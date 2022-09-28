import freetensor as ft
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--addr",
                    type=str,
                    help="known_server_ip",
                    default="127.0.0.1")
parser.add_argument("--port", type=int, help="known_server_port", default=8047)
parser.add_argument("--sev_status",
                    help="the_queue_of+\
this_server, use this flag more than once+\
 to add more queues, each at a time",
                    action="append",
                    default=["default"])

parser.add_argument("--self_server_ip",
                    type=str,
                    help="self_server_ip",
                    default="localhost")

parser.add_argument("--self_server_port",
                    type=int,
                    help="self_server_port",
                    default=0)

parser.add_argument("--shutdown_timeout",
                    type=float,
                    help="shutdown when idle after timeout",
                    default=-1)

parser.add_argument('--disable_pex',
                    action='store_true',
                    help="import this flag to disable +\
pex. Warning: without pex, you may fail to discover some peers if connection lost"
                   )

args = parser.parse_args()
# addr = "127.0.0.1"
# port = 8047
# sev_status = ["default"]
# self_server_ip = "localhost"
# self_server_port = 0
# shutdown_idle_timeout = -1

client = ft.MultiMachineScheduler(addr=args.addr,
                                  port=args.port,
                                  sev_status=args.sev_status,
                                  self_server_ip=args.self_server_ip,
                                  self_server_port=args.self_server_port,
                                  is_auto_pex=not (args.disable_pex))
client.rpctool.server_auto_shutdown(args.shutdown_timeout)

# the followings are what default default config
# client = ft.MultiMachineScheduler(addr= addr, port= port, sev_status= sev_status,
#                             self_server_ip= self_server_ip, self_server_port= self_server_port)
# client.server_auto_shutdown(shutdown_idle_timeout)
