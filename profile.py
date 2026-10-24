import geni.portal as portal
import geni.rspec.pg as rspec
import geni.rspec.emulab

request = portal.context.makeRequestRSpec()
lan = request.LAN("lan")

ubuntu_image = (
    "urn:publicid:IDN+emulab.net+image+"
    "emulab-ops//UBUNTU24-64-STD"
)

for name, ip in (("node0", "10.10.1.1"), ("node1", "10.10.1.2")):
    node = request.RawPC(name)
    node.hardware_type = "d7525"
    node.disk_image = ubuntu_image

    iface = node.addInterface("if0")
    iface.addAddress(rspec.IPv4Address(ip, "255.255.255.0"))
    lan.addInterface(iface)

    storage = node.Blockstore("docker_data_" + name, "/mydata")
    storage.size = "100GB"

portal.context.printRequestRSpec()
