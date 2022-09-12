import freetensor as ft
import sys

config = ["127.0.0.1", 8047, ["default"], "localhost", 0, -1]

for i in range(1, len(sys.argv)):
    config[i - 1] = sys.argv[i]

# addr = "127.0.0.1"
# port = 8047
# sev_status = ["default"]
# self_server_ip = "localhost"
# self_server_port = 0
# shutdown_idle_timeout = -1

client = ft.MultiMachineScheduler(*(config[0:5]))
client.rpctool.server_auto_shutdown(config[5])

# the followings are what default default config
# client = ft.MultiMachineScheduler(addr= addr, port= port, sev_status= sev_status,
#                             self_server_ip= self_server_ip, self_server_port= self_server_port)
# client.server_auto_shutdown(shutdown_idle_timeout)
