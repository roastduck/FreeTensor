from email.header import make_header
from sre_constants import SUCCESS
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client as client
import sys, socket, uuid, os, time
import threading

List = {}

if os.path.exists('./machine_list'):
    os.remove('./machine_list')


def check_connection():
    return True


def register_machine(remoteInfo, sev_status):
    """机器注册函数，从每个机器拉取网络地址和端口并且分配uuid，同时更新每个机器的本地列表"""
    global List
    print("Registering %s:%d" % (remoteInfo[0], remoteInfo[1]))
    UID = str(uuid.uuid4())
    try:
        with open('./machine_list', 'a') as fileList:
            fileList.write(remoteInfo[0] + ',' + str(remoteInfo[1]) + ',' +
                           UID + '\n')  #用uuid4函数分配随机不同的uuid
            fileList.close
    except IOError:
        print(
            "Error occured when creating or writing into the FILE of MACHINE LIST"
        )

    List[UID] = [remoteInfo[0], remoteInfo[1], sev_status]
    broadcast(UID, sev_status, new_tag=True)
    remote_server = connect(remoteInfo)
    for uid, info in List.items():
        if uid != UID:
            remote_server.change_status(uid, info[2], True)
    return str(UID)


def connect(addr):
    if "http" not in addr[0]:
        addr[0] = "http://" + addr[0]
    for cnt in range(5):
        try:
            server = client.ServerProxy(str(addr[0]) + ':' + str(addr[1]))
        except:
            time.sleep(0.1)
            continue
        break
    return server


def broadcast(host_uid, status, new_tag=False):
    """向所有机器广播host_uid的状态变更"""
    if status == 0:
        del List[host_uid]
    else:
        List[host_uid][2] = status
    for uid, addr in List.items():
        remote_server = connect(addr)
        remote_server.change_status(host_uid, status, new_tag)


def task_submit(remote_host_uid, src_host_uid, task):
    remote_host_uid = str(remote_host_uid)
    if remote_host_uid not in List:
        return -1
    remote_server = connect(List[remote_host_uid])
    status = remote_server.remote_task_receive(src_host_uid, task)
    broadcast(remote_host_uid, status)
    return status


def result_submit(remote_host_uid, src_host_uid, task_result):
    remote_host_uid = str(remote_host_uid)
    if remote_host_uid not in List:
        return -1
    remote_server = connect(List[remote_host_uid])
    return remote_server.remote_result_receive(src_host_uid, task_result)

def run_center():
# 获取内网IP和可用端口
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        SocketName = (s.getsockname()[0], 8047)  #初始化端口为8047，需要手动用ufw开防火墙端口
        s.close()

    server = SimpleXMLRPCServer(SocketName, allow_none=True)
    print("Receiving Message on %s:%d..." % (SocketName[0], SocketName[1]))
    server.register_function(check_connection)
    server.register_function(register_machine)
    server.register_function(task_submit)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if os.path.exists('./machine_list'):
            os.remove('./machine_list')
        print("\nKeyboard interrupt received, exiting.")
        sys.exit(0)
