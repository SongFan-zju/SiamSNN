import socket
import struct


class Transmitter(object):

    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()

    def send_flit_bin(self, flit_bin_file, data_type):
        '''
        发送flit
        '''
        with open(flit_bin_file, 'rb') as file:
            flit_bin = file.read()
        length = len(flit_bin) >> 2
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_bin_file)
            print("===<2>=== send flit length failed")
            return 0
        send_bytes = bytearray()
        send_bytes += struct.pack('I', length)
        send_bytes += struct.pack('I', data_type)
        send_bytes += flit_bin
        #self.socket_inst.sendall(struct.pack('I', length))
        #ack = self.socket_inst.recv(1024)
        #if (ack == b'done'):
        #    print("===<2>=== send flit length succeed")
        #self.socket_inst.sendall(flit_bin)
        self.socket_inst.sendall(send_bytes)
        return 1

    def send_flit(self, flit_file, directions=0):
        '''
        发送flit
        '''
        with open(flit_file, 'r') as file:
            flit_list = file.readlines()
        length = len(flit_list)
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_file)
            print("===<2>=== send flit length failed")
            return 0
        # print("===<2>=== send flit length succeed")
        #self.socket_inst.sendall(struct.pack('I', length))
        #ack = self.socket_inst.recv(1024)
        #if (ack == b'done'):

        j = 0
        while (j < length):
            send_bytes = bytearray()
            send_bytes += struct.pack('Q', length)
            # send_bytes += struct.pack('I', 0x8000) # TODO tik
            for i in range(j, min(j + 16777216 * 4, length)):
                send_bytes += struct.pack('Q', int(flit_list[i % length].strip(), 16))
            self.socket_inst.sendall(send_bytes)
            j = j + 16777216 * 4
            if (j <= length):
                reply = self.socket_inst.recv(1024)
                print("%s" % reply)
        return 1
