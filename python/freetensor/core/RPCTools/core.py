import xmlrpc.client as client
from multiprocessing import Process, Pool
from xmlrpc.server import SimpleXMLRPCServer
import socket, sys, time


class RPCTool:
    def __init__(self, addr, port = 8047): #参数是初始化时主服务器的地址
        """初始化获取主机地址和端口以便分配UUID"""
        self.serverAddr = self.get_address()
        self.server = SimpleXMLRPCServer(self.serverAddr, allow_none = True)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_function(remote_task_receive)
        self.server.register_function(remote_result_receive)
        self.server.register_function(add_host)
        self.server.register_function(remove_host)
        self.server.register_function(self.change_status)
        try:
            self.serverProcess = Process(target = self.server.serve_forever)
            self.serverProcess.start()
            print("RPC Server Started on %s:%d..." % (self.serverAddr[0], self.serverAddr[1]))
            self.upload([addr, port])
            get_self_uid(self.UID)
            print("Machine UID:" + self.UID)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

    def connect(self, addr):
        """允许失败五次的连接，每次连接之间间隔0.5s"""
        if "http" not in addr[0]:
            addr[0] = "http://" + addr[0]
        for cnt in range(5):
            try:
                server = client.ServerProxy(str(addr[0]) + ':' + str(addr[1]))
            except:
                time.sleep(0.5)
                continue
            break
        return server

    def get_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        s.connect(('8.8.8.8', 80))
        Addr = s.getsockname()
        s.close()
        return Addr[0], Addr[1]

    def upload(self, addr):
        """上传当前服务器信息给中心服务器"""
        try:
            self.center_server = self.connect(addr)
            print("Server Connected\nRegistering machine...")
            self.UID = self.center_server.register_machine(self.serverAddr)
            print("Registered")
        except Exception as Er:
            print("Register Failed")
            print(Er)
            sys.exit(1)

    def change_status(self, remote_host_uid, status, new_tag = False):
        print("Status Changed:" + remote_host_uid + " " + str(status))
        if new_tag == False:
            remove_host(remote_host_uid)
        if status:
            add_host(remote_host_uid, status)

    def remote_task_submit(self, remote_host_uid, task):
        return self.center_server.task_submit(remote_host_uid, self.UID, task)

    def remote_result_submit(self, remote_host_uid, task_result):
        return self.center_server.result_submit(remote_host_uid, self.UID, task_result)

# def remove_host(uid):
#     print("Host %s removed" % uid)
# def add_host(uid, status):
#     print("Added a host %s with status %s" % (uid, str(status)))