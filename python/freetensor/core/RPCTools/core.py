import xmlrpc.client as client
from xmlrpc.server import SimpleXMLRPCServer
import socket, time
from .. import remote_task_scheduler
import threading


class RPCTool:

    def __init__(
            self,
            remoteTaskScheduler: remote_task_scheduler.RemoteTaskScheduler,
            center_addr='127.0.0.1',
            center_port=8047,
            sev_status=3):  #fill the address and port with your center server's
        """Initially get the address,port and uid of this node"""
        self.UID = 'localhost'  #the node will run locally with a special uid 'localhost' when there's no center server
        self.taskScheduler = remoteTaskScheduler
        self.serverAddr = self.get_address(center_addr, center_port)
        self.server = SimpleXMLRPCServer(self.serverAddr, allow_none=True)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_function(self.taskScheduler.remote_task_receive)
        self.server.register_function(self.taskScheduler.remote_result_receive)
        self.server.register_function(self.change_status)
        self.server.register_function(self.check_connection)
        self.server.register_function(self.quit)
        try:
            self.serverThread = threading.Thread(target=self.serve_till_quit)
            self.serverThread.start()
            print("RPC Server Started on %s:%d..." %
                  (self.serverAddr[0], self.serverAddr[1]))
            self.centerAddr = [center_addr, center_port]
            self.upload(self.centerAddr, sev_status)
            self.taskScheduler.get_self_uid(self.UID)
            print("Machine UID:" + self.UID)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

    def serve_till_quit(self):
        self.quit_tag = False
        while not self.quit_tag:
            self.server.handle_request()

    def quit(self):
        self.quit_tag = True
        client.ServerProxy('http://' + self.serverAddr[0] + ':' +
                           str(self.serverAddr[1]))

    def connect(self, addr):
        """The number of failed connections can be allowed to be 5 at most."""
        if "http" not in addr[0]:
            addr[0] = "http://" + addr[0]
        for cnt in range(5):
            try:
                server = client.ServerProxy(str(addr[0]) + ':' + str(addr[1]))
                server.check_connection()
            except Exception as Ex:
                time.sleep(0.1)
                server = None
                continue
            except SystemExit:
                pass
            break
        if server:
            return server
        else:
            raise Exception("Error failed to connect")

    def get_address(self, addr, port):
        """This function is used to get this node's IP address and an available port"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        s.connect((addr, port))
        Addr = s.getsockname()
        s.close()
        return Addr[0], Addr[1]

    def upload(self, addr, sev_status):
        """Upload the server's address to the center server and get an uid from it."""
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
            return self.taskScheduler.remote_task_receive(self.UID, task)
        center_server = self.connect(self.centerAddr)
        return center_server.task_submit(remote_host_uid, self.UID, task)

    def remote_result_submit(self, remote_host_uid, task_result):
        if remote_host_uid == "localhost":
            return self.taskScheduler.remote_result_receive(
                self.UID, task_result)
        center_server = self.connect(self.centerAddr)
        return center_server.result_submit(remote_host_uid, self.UID,
                                           task_result)
