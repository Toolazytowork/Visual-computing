import sys
sys.path.append('.')

from servers.base import Server
import argparse

class MockArgs:
    def __init__(self):
        self.trainer = argparse.Namespace(num_clients=10)

args = MockArgs()
server = Server(args)

print("Initial selection (top 5 from 10, should be first 5 as all are inf):")
print(server.select_clients(5))

print("\nUpdating losses...")
server.update_loss(0, 5.0)
server.update_loss(1, 10.0)
server.update_loss(2, 2.0)
server.update_loss(3, 15.0)
server.update_loss(4, 1.0)
# Clients 5-9 are still inf

print("\nSecond selection (top 5 from 10, expecting [5, 6, 7, 8, 9] since they are inf):")
print(server.select_clients(5))

print("\nUpdating more losses to remove all inf...")
server.update_loss(5, 7.0)
server.update_loss(6, 12.0)
server.update_loss(7, 3.0)
server.update_loss(8, 0.5)
server.update_loss(9, 20.0)

print("\nThird selection (top 5 from 10, expecting [9, 3, 6, 1, 5] as top losses are 20, 15, 12, 10, 7):")
print(server.select_clients(5))
