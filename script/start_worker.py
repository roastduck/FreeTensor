import freetensor as ft
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gatewayaddr",
                    type=str,
                    help="known_server(gateway)_ip",
                    default="127.0.0.1")
parser.add_argument("--gatewayport",
                    type=int,
                    help="known_server(gateway)_port",
                    default=8047)
parser.add_argument("--sevstatus",
                    help="the_queue_of\
this_server, use this flag more than once\
 to add more queues, each at a time",
                    action="append",
                    default=["default"])

parser.add_argument("--ip",
                    type=str,
                    help="self_server_ip",
                    default="0.0.0.0")

parser.add_argument("--port", type=int, help="self_server_port", default=0)

parser.add_argument("--timeout",
                    type=float,
                    help="shutdown when idle after timeout",
                    default=-1)

parser.add_argument('--disable_pex',
                    action='store_true',
                    help="import this flag to disable \
pex. Warning: without pex, you may fail to discover some peers if connection lost"
                   )

args = parser.parse_args()

client = ft.MultiMachineScheduler(addr=args.gatewayaddr,
                                  port=args.gatewayport,
                                  sev_status=args.sevstatus,
                                  self_server_ip=args.ip,
                                  self_server_port=args.port,
                                  is_auto_pex=not (args.disable_pex))
client.rpctool.server_auto_shutdown(args.timeout)
