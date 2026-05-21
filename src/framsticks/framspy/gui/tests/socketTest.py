import unittest

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

from gui.framsutils.FramsSocket import FramsSocket

class SocketTest(unittest.TestCase):
    # Make sure to launch an empty server, without any running jobs etc.
    ADDRESS = "127.0.0.1"
    PORT = 9009

    def test_connection(self):
        frams = FramsSocket()
        
        try:
            frams.initConnection(self.ADDRESS, self.PORT)
            self.assertTrue(frams.comm.client.connected, "Connection inner state")
        except ConnectionError:
            self.assertTrue(False, "ConnectionError exception")

        frams.closeConnection()

    def test_communication_parsing1(self):
        frams = FramsSocket()
        frams.initConnection(self.ADDRESS, self.PORT)

        response = frams.sendRequest("get / motd")
        data = frams._infoParser(response)
        frams.closeConnection()

        self.assertEqual(len(data), 1, "Response is to short")
        self.assertEqual(data[0].p["_classname"], "Server")
        self.assertIn("motd", data[0].p) #check if contains motd field
        self.assertIsInstance(data[0].p["motd"], str) #check if response have a corect type

    def test_communication_parsing2(self):
        frams = FramsSocket()
        frams.initConnection(self.ADDRESS, self.PORT)

        response = frams.sendRequest("get /simulator{version_string,version_int}")
        data = frams._infoParser(response)
        frams.closeConnection()

        self.assertEqual(len(data), 1, "Response is to short")
        self.assertEqual(data[0].p["_classname"], "Simulator")
        self.assertIn("version_string", data[0].p) #check if contains motd field
        self.assertIsInstance(data[0].p["version_string"], str) #check if response have a corect type
        self.assertIn("version_int", data[0].p) #check if contains motd field
        self.assertIsInstance(data[0].p["version_int"], int) #check if response have a corect type

    def test_genotype_mod(self):
        frams = FramsSocket()
        frams.initConnection(self.ADDRESS, self.PORT)

        genotype = "X"
        info = "test info"

        frams.sendRequest("call /simulator/genepools/groups/0 clear")
        frams.sendRequest("call /simulator/genepools/groups/0 add {}".format(genotype))
        genotypeInfo = frams.readGenotypeInfo(0, 0)
        prop = next((p for p in genotypeInfo if p.p["id"] == "genotype"), None)
        self.assertTrue(prop, "genotype property not found")
        self.assertEqual(prop.p["value"], genotype, "genotype not match")
        
        frams.sendRequest("set /simulator/genepools/groups/0/genotypes/0 info \"{}\"".format(info))
        genotypeInfo = frams.readGenotypeInfo(0, 0)
        prop = next((p for p in genotypeInfo if p.p["id"] == "info"), None)
        self.assertTrue(prop, "info property not found")
        self.assertEqual(prop.p["value"], info, "info not match")

        frams.sendRequest("call /simulator/genepools/groups/0/genotypes/0 delete")

        frams.closeConnection()

    def test_multiline_strings(self):
        frams = FramsSocket()
        frams.initConnection(self.ADDRESS, self.PORT)

        info = r"test\ntest2"
        infoResponse = "test\ntest2"

        #prepare known multiline field
        frams.sendRequest("call /simulator/genepools/groups/0 clear")
        frams.sendRequest("call /simulator/genepools/groups/0 add {}".format("X"))

        #set and get multiline field
        frams.sendRequest("set /simulator/genepools/groups/0/genotypes/0 info \"{}\"".format(info))
        genotypeInfo = frams.readGenotypeInfo(0, 0)
        prop = next((p for p in genotypeInfo if p.p["id"] == "info"), None)
        self.assertTrue(prop, "info property not found")
        self.assertMultiLineEqual(prop.p["value"], infoResponse, "multiline string not match")

        frams.sendRequest("call /simulator/genepools/groups/0/genotypes/0 delete")

        frams.closeConnection()

    def test_multiline_string_encoding(self):
        import gui.framsutils.framsProperty as framsProperty

        frams = FramsSocket()
        frams.initConnection(self.ADDRESS, self.PORT)

        #prepare known multiline field
        frams.sendRequest("call /simulator/genepools/groups/0 clear")
        frams.sendRequest("call /simulator/genepools/groups/0 add {}".format("X"))

        info = \
r"Long multiline text\
1\\~\
22\~\n\
333\ \\\
4444"
        encodedInfo = framsProperty.encodeString(info)

        frams.sendRequest("set /simulator/genepools/groups/0/genotypes/0 info \"{}\"".format(encodedInfo))
        genotypeInfo = frams.readGenotypeInfo(0, 0)
        prop = next((p for p in genotypeInfo if p.p["id"] == "info"), None)
        self.assertTrue(prop, "info property not found")
        self.assertMultiLineEqual(prop.p["value"], info, "encoded string not match")

        frams.sendRequest("call /simulator/genepools/groups/0/genotypes/0 delete")

        frams.closeConnection()

if __name__ == "__main__":
    unittest.main()