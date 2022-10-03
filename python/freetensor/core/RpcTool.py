from __future__ import annotations
import socket
import threading
import socketserver
import time
import struct
import multiprocessing
from typing import Dict, List
import pickle
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import random
from .. import core
import sys


#rewrite send/recv to ensure stable data transfer
def send_msg(sock: socket.socket, msg):
    # Prefix each message with a 8-byte length (network byte order)
    msg = struct.pack('!Q', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock: socket.socket, timeout=30):
    # Read message length and unpack it into an unsigned long long
    sock.settimeout(max(timeout, 5))
    raw_msglen = recvall(sock, 8)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!Q', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock: socket.socket, n, default_retries=4):
    # Helper function to recv n bytes or raise Timeout Exception
    fragments = []
    tries = default_retries
    while True:
        chunk = sock.recv(min(n, 4096))
        t = len(chunk)
        n -= t
        if n <= 0:
            if t > 0:
                fragments.append(chunk)
            break
        if t > 0:
            fragments.append(chunk)
            tries = default_retries
        else:
            time.sleep(0.5)
            tries -= 1  #waiting for data to come
            if tries < 0:
                raise TimeoutError("receive time out")
    return b''.join(fragments)


class RPCToolThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    rpctool: RPCTool

    def __init__(self, request, client_address, server, rpctool) -> None:
        self.rpctool = rpctool
        super().__init__(request, client_address, server)

    def handle(self):
        data = self.rpctool.deserialize(recv_msg(self.request))
        self.rpctool.response(self.request, data)
        self.finish()


class RPCToolThreadedTCPServer(socketserver.ThreadingMixIn,
                               socketserver.TCPServer):
    rpctool: RPCTool

    def __init__(self, server_address, RequestHandlerClass,
                 rpctool: RPCTool) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.rpctool = rpctool

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass."""
        try:
            self.RequestHandlerClass(request, client_address, self,
                                     self.rpctool)
        except Exception:
            pass


'''
the message contains two parts:
key: "function": 
    value:
        "ping": check connection and availability, return success if received(with address attatched)
            "host_uid": str
            "target_uid": str     #this is implemented to remove revoked uid

        "join": join the network
            "host_uid": str
            "server_addr": tuple[str, int] #the address of gateway

        "exit": leave the network gracefully
            "host_uid": str

        "pex_update": get hosts list(with ip and port)
            "server_info_list": Dict[str, tuple] (for example, {host_uid: address} )

        Note: using PEX, we can quickly find adquate hosts without bothering center.

        "exchange_status": exchange the status of hosts
            "host_uid": str
            "target_uid": str     #this is implemented to remove revoked uid
            "address": tuple[str, int]  src address
            "last_modify_time": float   src last modify time
            "sev_status": List  src sev_status
        
        "transfer": transfer data
            "host_uid": str
            "target_uid": str     #this is implemented to remove revoked uid
            "type": str
            "data": the data to send(scheduler defined)

        "return": put back return values(generally the connection statys)
            "return_status": str               
'''


class RPCTool(object):
    host_list: Dict[str, Dict]
    host_list_lock = threading.RLock()
    scheduler_host_list: set
    scheduler_host_list_lock: threading.RLock = threading.RLock()

    self_server: RPCToolThreadedTCPServer
    self_host_uid: str
    scheduler: core.RemoteTaskScheduler
    centre_address: tuple[str, int]
    sev_status: List
    self_and_gateway_addr: tuple[tuple[str, int], tuple[str, int]]

    verbose = 1
    last_active_time: float
    last_modify_time: float
    is_center: bool
    max_connection: int = 200
    max_reconnect_retries: int = 5
    pool: ThreadPoolExecutor
    server_closed = False

    def __init__(self,
                 scheduler=None,
                 host: str = "localhost",
                 port: int = 8047,
                 sev_status: List[str] = ["default"],
                 is_center=True,
                 self_server_port=0,
                 self_server_ip="localhost",
                 is_auto_pex=True) -> None:
        HOST, PORT = self_server_ip, self_server_port
        server = RPCToolThreadedTCPServer(
            (HOST, PORT), RPCToolThreadedTCPRequestHandler, self)
        self.self_server = server
        self.pool = ThreadPoolExecutor()
        self.scheduler_host_list = set()
        self.host_list = {}
        self.server_closed = False

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        if self.verbose > 0:
            print("Server loop running in thread:", server_thread.name,
                  self.self_server.server_address)

        self.last_active_time = time.time()
        self.scheduler = scheduler
        if host == "None":
            self.centre_address = None
            self.self_and_gateway_addr = None
        else:
            self.centre_address = (host, port)
            self.self_and_gateway_addr = self.get_ips(self.centre_address)
        self.sev_status = sev_status
        self.is_center = is_center
        self.internet_init()
        #self.scheduler.get_self_uid(self.self_host_uid)
        if is_auto_pex:
            self.server_auto_pex(15)

    def get_ips(self, gateway_addr: tuple[str, int]):
        tries = 5
        while (tries > 0):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect(gateway_addr)
                return ((sock.getsockname()[0],
                         self.self_server.server_address[1]),
                        sock.getpeername())
            except Exception:
                tries -= 1
        return None

    def remote_task_submit(self, server_uid: str, task: Dict):
        if self.server_closed:
            return -1
        tmpdict = {
            "function": "transfer",
            "host_uid": self.self_host_uid,
            "target_uid": server_uid,
            "type": "task",
            "data": task
        }
        ret = self.send(self.get_address(server_uid), tmpdict, 20)
        if ret["return_status"] == 0:
            return 0
        else:
            return -1

    def remote_result_submit(self, server_uid: str, taskresult: Dict):
        if self.server_closed:
            return
        tmpdict = {
            "function": "transfer",
            "host_uid": self.self_host_uid,
            "target_uid": server_uid,
            "type": "taskresult",
            "data": taskresult
        }
        self.send(self.get_address(server_uid), tmpdict, -1)

    def transfer_recv(self, sock: socket.socket, data: Dict):
        if data["target_uid"] == self.self_host_uid:
            if data["type"] == "task":
                ret = self.scheduler.remote_task_receive(
                    data["host_uid"], data["data"])
                tmpdict = {"function": "return", "return_status": ret}
                self.send_with_retries(sock, tmpdict, -1)
            elif data["type"] == "taskresult":
                self.scheduler.remote_result_receive(data["host_uid"],
                                                     data["data"])

    def scheduler_list_auto_append(self, server_uid: str):
        self.scheduler_host_list_lock.acquire()
        if server_uid in self.scheduler_host_list:
            self.scheduler_host_list_lock.release()
        else:
            self.scheduler_host_list.add(server_uid)
            self.scheduler_host_list_lock.release()
        self.pool.submit(self.scheduler_host_add, server_uid)

    def scheduler_list_auto_remove(self, server_uid: str):
        self.scheduler_host_list_lock.acquire()
        if server_uid in self.scheduler_host_list:
            self.scheduler_host_list.discard(server_uid)
            self.scheduler_host_list_lock.release()
            try:
                t = threading.Thread(target=self.scheduler_host_remove,
                                     args=(server_uid,))
                t.daemon = True
                t.start()
                # self.pool.submit(self.scheduler_host_remove, server_uid)
            except Exception:
                pass
        else:
            self.scheduler_host_list_lock.release()

    def scheduler_host_add(self, server_uid: str):
        if not (self.scheduler is None):
            if not (server_uid in self.scheduler.server_list):
                sev_dict = self.host_list.get(server_uid, {})
                self.scheduler.add_host(server_uid,
                                        sev_dict.get("sev_status", ["default"]))

    def scheduler_host_remove(self, server_uid: str):
        if not (self.scheduler is None):
            self.scheduler.remove_host(server_uid)

    def serialize(self, tmpdict: Dict):
        return pickle.dumps(tmpdict)

    def deserialize(self, tmpstr: str):
        return pickle.loads(tmpstr)

    def send_with_retries(self, sock: socket.socket, message: Dict, timeout=30):
        ret_val = {"function": "return", "return_status": "fail"}
        sock.settimeout(max(timeout, 5))
        for i in range(3):
            try:
                send_msg(sock, self.serialize(message))
                if timeout <= 0:
                    sock.shutdown(2)
                    # sock.close()
                    return None
                else:
                    ret_val = self.deserialize(recv_msg(sock, timeout))
                    sock.shutdown(2)
                    # sock.close()
            except Exception as e:
                time.sleep(1)
                print(e)
                print(message)
            else:
                break
        return ret_val

    def get_socket(self, host_address):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(host_address)
        return sock

    def send_base(self,
                  host_address,
                  raw_message: Dict,
                  pipe: multiprocessing.Pipe,
                  timeout=30):
        if host_address is None:
            pipe.send({"function": "return", "return_status": "fail"})
            return
        else:
            try:
                sock: socket.socket = self.get_socket(host_address)
                pipe.send(self.send_with_retries(sock, raw_message, timeout))
                # sock.close()
            except Exception:
                pipe.send({"function": "return", "return_status": "fail"})
        sys.exit(0)

    def send(self, host_address, raw_message: Dict, timeout=30):
        send_pipe, recv_pipe = multiprocessing.Pipe()
        t = multiprocessing.Process(target=self.send_base,
                                    args=(host_address, raw_message, send_pipe,
                                          timeout))
        t.daemon = True
        t.start()
        t.join()
        return recv_pipe.recv()

    def response(self, sock: socket.socket, data: Dict):
        self.last_active_time = time.time()
        if not ("function" in data):
            return
        task_type = data["function"]
        if self.verbose > 0:
            print(self.self_host_uid, " ", task_type)
        if task_type == "ping":
            self.pingrecv(sock, data)
        elif task_type == "exchange_status":
            self.exchange_status_recv(sock, data)
        elif task_type == "pex_update":
            self.update_status(data["server_info_list"])
        elif task_type == "exit":
            self.remove_host(data["host_uid"])
        elif task_type == "join":
            self.join_recv(sock, data)
        elif task_type == "transfer":
            self.transfer_recv(sock, data)

    def pex_update_base(self, server_list: List[str]):
        if not server_list:
            server_list = list(self.host_list)
        server_info_list = {key: self.get_address(key) for key in server_list}
        return {"function": "pex_update", "server_info_list": server_info_list}

    def check_uid_existence(self, uid: str):
        return (uid in self.host_list)

    def join_recv(self, sock: socket.socket, data: Dict):
        if self.check_uid_existence(data["host_uid"]):
            tmpdict = {"function": "return", "return_status": "not_available"}
        else:
            if self.self_and_gateway_addr is None:
                self.host_list[
                    self.self_host_uid]["address"] = data["server_addr"]
                self.self_and_gateway_addr = (
                    data["server_addr"],
                    None,
                )
            tmpdict = self.pex_update_base([])
        self.send_with_retries(sock, tmpdict, -1)

    def get_address(self, host_uid: str):
        tmpdict = self.host_list.get(host_uid, {})
        return tmpdict.get("address", None)

    def ping(self, host_uid):
        address = self.get_address(host_uid)
        message = {
            "function": "ping",
            "host_uid": self.self_host_uid,
            "target_uid": host_uid
        }
        tmpdict: Dict = self.send(address, message, 5)
        if tmpdict["return_status"] == "success":
            self.last_active_time = time.time()
            self.refresh_host(host_uid)
            if self.verbose > 0:
                print("ping : ", host_uid, " success")
            return True
        else:
            if tmpdict["return_status"] == "not_available":
                self.remove_host(host_uid)
            else:
                self.remove_host_gracefully(host_uid)
            if self.verbose > 0:
                print("ping : ", host_uid, " failed")
            return False

    def pingrecv(self, sock: socket.socket, data: Dict):
        if self.server_closed == True:
            return
        if (data["target_uid"] == self.self_host_uid):
            tmpdict = {"function": "return", "return_status": "success"}
        else:
            tmpdict = {"function": "return", "return_status": "not_available"}
        self.send_with_retries(sock, tmpdict, -1)

    def internet_init(self):
        response: Dict
        for i in range(5):
            try:
                self.self_host_uid = str(uuid4())
                self.last_modify_time = time.time()
                tmpdict = {
                    "function": "join",
                    "host_uid": self.self_host_uid,
                    "server_addr": self.self_and_gateway_addr[1]
                }
                response = self.send(self.centre_address, tmpdict, 10)
            except Exception:
                time.sleep(1)
            else:
                if response["function"] == "pex_update":
                    self_info = {
                        "address": self.self_and_gateway_addr[0],
                        "last_modify_time": self.last_modify_time,
                        "sev_status": self.sev_status
                    }
                    self.host_list[self.self_host_uid] = self_info
                    self.update_status(response["server_info_list"])
                    return
                else:
                    pass
        if self.verbose > 0:
            print("initialize failed, now running locally at ",
                  self.self_server.server_address)
        self_info = {
            "address": self.self_server.server_address,
            "last_modify_time": self.last_modify_time,
            "sev_status": self.sev_status
        }
        self.host_list[self.self_host_uid] = self_info

    def update_status_single_base(self, host_uid):
        tmpdict = {
            "function": "exchange_status",
            "host_uid": self.self_host_uid,
            "target_uid": host_uid,
            "address": (self.self_and_gateway_addr[0]),
            "last_modify_time": self.last_modify_time,
            "sev_status": self.sev_status
        }
        return tmpdict

    def update_status_single_dict(self, host_uid: str, ret: Dict):
        if ret["function"] == "exchange_status":
            if ret["target_uid"] == self.self_host_uid:
                if ret["host_uid"] == "":
                    return False
                mykeys = {"address", "last_modify_time", "sev_status"}
                if type(ret["sev_status"]) == str:
                    ret["sev_status"] = [ret["sev_status"]]
                host_info = {}
                for key in mykeys:
                    host_info[key] = ret[key]
                self.host_list[host_uid] = host_info
                self.refresh_host(host_uid)
                self.last_active_time = time.time()
                self.scheduler_list_auto_append(host_uid)
                return True
            else:
                return False
        else:
            if ret["return_status"] == "not_available":
                self.remove_host(host_uid)
            else:
                self.remove_host_gracefully(host_uid)
            return False

    def update_status_single(self, host_uid: str, host_info: Dict):
        if host_uid == self.self_host_uid:
            return
        if self.verbose > 0:
            print("updating : ", host_uid, host_info)
        msg = self.update_status_single_base(host_uid)
        ret = self.send(host_info, msg, 10)
        self.update_status_single_dict(host_uid, ret)

    def update_status(self, host_list: Dict[str, Dict]):
        # pool = ThreadPoolExecutor(3)
        for i in list(host_list):
            try:
                self.pool.submit(self.update_status_single, i, host_list[i])
            except Exception:
                pass

    def exchange_status_recv(self, sock: socket.socket, data: Dict):
        if self.server_closed == True:
            return
        elif self.update_status_single_dict(data.get("host_uid", ""), data):
            tmpdict = self.update_status_single_base(data["host_uid"])
        else:
            tmpdict = {"function": "return", "return_status": "not_available"}
        self.send_with_retries(sock, tmpdict, -1)

    def try_connect_all(self):
        server_list = list(self.host_list)
        for t in server_list:
            if t == self.self_host_uid:
                continue
            if self.ping(t):
                return True
        return False

    def server_auto_shutdown(self, timeout: float = 5.0):
        time.sleep(10)
        if timeout <= 0.0:
            while (1):
                time.sleep(100)

        def server_auto_shutdown_base(self: RPCTool, timeout: float):
            while True:
                if time.time() > (timeout + self.last_active_time):
                    if self.try_connect_all():
                        pass
                    else:
                        break
                else:
                    time.sleep(1)
            self.end_server()

        # self.pool.submit(server_auto_shutdown_base, self, timeout)
        t = threading.Thread(target=server_auto_shutdown_base,
                             args=(self, timeout))
        # t.daemon = True
        t.start()

    def server_auto_pex(self, interval: float = 15.0):

        def server_auto_pex_base(self: RPCTool, interval: float):
            pex_max_counts: int = 5
            while (self.server_closed == False):
                time.sleep(interval + random.random() * 5)
                t = list(self.host_list)
                if t:
                    pex_list = random.sample(t, min(len(t), pex_max_counts))
                    tmpdict = self.pex_update_base(pex_list)
                    self.send(self.get_address(random.choice(pex_list)),
                              tmpdict, -1)
            print("pex end")

        t = threading.Thread(target=server_auto_pex_base, args=(self, interval))
        t.daemon = True
        t.start()
        # self.pool.submit(server_auto_pex_base, self, interval)

    def remove_host(self, host_uid: str):
        self.host_list_lock.acquire()
        try:
            self.host_list.pop(host_uid)
        except KeyError:
            pass
        self.host_list_lock.release()
        self.scheduler_list_auto_remove(host_uid)

    def remove_host_gracefully(self, host_uid: str):
        self.host_list_lock.acquire()
        try:
            if host_uid in self.host_list:
                self.host_list[host_uid].setdefault("tries",
                                                    self.max_reconnect_retries)
                self.host_list[host_uid]["tries"] -= 1
                if self.host_list[host_uid]["tries"] < 0:
                    self.remove_host(host_uid)
        except Exception:
            pass
        self.host_list_lock.release()

    def refresh_host(self, host_uid: str):
        try:
            self.host_list[host_uid]["tries"] = self.max_reconnect_retries
        except Exception:
            pass

    def end_server(self):
        self.self_server.shutdown()
        self.server_closed = True
        if self.verbose > 0:
            print("server_shutting down : ", self.self_server.server_address)
        message = {"function": "exit", "host_uid": self.self_host_uid}
        # pool = ThreadPoolExecutor(4)
        host_list = list(self.host_list)
        for i in host_list:
            if i == self.self_host_uid:
                continue
            try:
                self.pool.submit(self.send, self.get_address(i), message, -1)
            except Exception:
                pass
            # self.send(self.get_address(i),message, -1)
        for i in host_list:
            self.remove_host(i)
        if self.verbose > 0:
            print("request sent")
        self.pool.shutdown(wait=True, cancel_futures=True)
        del self.scheduler
