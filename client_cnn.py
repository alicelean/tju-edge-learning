import socket
import time
import struct
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))