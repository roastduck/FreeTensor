import xmlrpc.client as client
from multiprocessing import Process, Pool
from xmlrpc.server import SimpleXMLRPCServer
import socket, sys, time
from .. import remote_task_scheduler


class RPCTool:

    def __init__(self,
                 remoteTaskScheduler: remote_task_scheduler.RemoteTaskScheduler,
                 addr="127.0.0.1",
                 port=8047,
                 sev_status=3):  #参数是初始化时主服务器的地址
        """初始化获取主机地址和端口以便分配UUID"""
        self.UID = 'localhost'  #当无中心服务器分配UID时默认本地运行，分配特殊UID:'localhost
        self.taskScheduler = remoteTaskScheduler
        self.serverAddr = self.get_address()
        self.server = SimpleXMLRPCServer(self.serverAddr, allow_none=True)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_function(self.taskScheduler.remote_task_receive)
        self.server.register_function(self.taskScheduler.remote_result_receive)
        self.server.register_function(self.change_status)
        self.server.register_function(self.check_connection)
        self.server.register_function(self.quit)
        try:
            self.serverProcess = Process(target=self.server.serve_forever)
            self.serverProcess.start()
            print("RPC Server Started on %s:%d..." %
                  (self.serverAddr[0], self.serverAddr[1]))
            self.centerAddr = [addr, port]
            self.upload(self.centerAddr, sev_status)
            self.taskScheduler.get_self_uid(self.UID)
            print("Machine UID:" + self.UID)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
        
    def quit(self):
        self.serverProcess.kill()
        print(str(self.UID) + " quited.")

    def connect(self, addr):
        """允许失败五次的连接，每次连接之间间隔0.1s"""
        if "http" not in addr[0]:
            addr[0] = "http://" + addr[0]
        for cnt in range(5):
            try:
                server = client.ServerProxy(str(addr[0]) + ':' + str(addr[1]))
                server.check_connection()
            except:
                time.sleep(0.1)
                server = None
                continue
            break
        if server:
            return server
        else:
            raise Exception("Error failed to connect")

    def get_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        s.connect(('8.8.8.8', 80))
        Addr = s.getsockname()
        s.close()
        return Addr[0], Addr[1]

    def upload(self, addr, sev_status):
        """上传当前服务器信息给中心服务器"""
        try:
            center_server = self.connect(addr)
            print("Server Connected\nRegistering machine...")
            self.UID = center_server.register_machine(self.serverAddr,
                                                      sev_status)
            print("Registered")
        except Exception as Er:
            print(Er)
            print("Remote registering failed, running locally")

    def check_connection(self):
        return True

    def change_status(self, remote_host_uid, status, new_tag=False):
        print("Status Changed:" + remote_host_uid + " " + str(status))
        if new_tag == False:
            self.taskScheduler.remove_host(remote_host_uid)
        self.taskScheduler.add_host(remote_host_uid, status)
    
    def quitcenter(self):
        center_server = self.connect(self.centerAddr)
        center_server.shutdown_center()

    def remote_task_submit(self, remote_host_uid, task):
        if remote_host_uid == "localhost":
            return self.taskScheduler.task_submit(remote_host_uid, self.UID,
                                                  task)
        center_server = self.connect(self.centerAddr)
        return center_server.task_submit(remote_host_uid, self.UID, task)

    def remote_result_submit(self, remote_host_uid, task_result):
        if remote_host_uid == "localhost":
            return self.taskScheduler.result_submit(remote_host_uid, self.UID,
                                                    task_result)
        center_server = self.connect(self.centerAddr)
        return center_server.result_submit(remote_host_uid, self.UID,
                                           task_result)


# def remove_host(uid):
#     print("Host %s removed" % uid)
# def add_host(uid, status):
#     print("Added a host %s with status %s" % (uid, str(status)))
# def task_submit(r_uid, l_uid, task):
#     print("Submit a task from %s to %s" % (r_uid, l_uid))
# def result_submit(r_uid, l_uid, result):
#     print("Submit result from %s to %s" % (r_uid, l_uid))
