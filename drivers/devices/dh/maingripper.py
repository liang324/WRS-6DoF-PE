import drivers.devices.dh.dh_modbus_gripper as dh
from time import sleep


class MainGripper(object):
    def __init__(self, port='com5', baudrate=115200, force=10, speed=30):
        port = port
        baudrate = baudrate
        initstate = 0
        force = force
        speed = speed
        self.m_gripper = dh.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)
        self.init_gripper()
        while (initstate != 1):
            initstate = self.m_gripper.GetInitState()
            sleep(0.2)
        self.m_gripper.SetTargetSpeed(speed)
        self.m_gripper.SetTargetForce(force)


    def init_gripper(self):
        self.m_gripper.Initialization()
        sleep(1)
        print('Send grip init')


    def mg_set_force(self, force):
        '''
        set the force for main gripper
        '''
        self.m_gripper.SetTargetForce(force)


    def mg_set_vel(self, vel):
        '''
        set the max vel for main gripper
        '''
        self.m_gripper.SetTargetSpeed(vel)

    def conv2encoder(self, jawwidth):
        a = int(jawwidth * 1000 / 0.076)
        return a

    def zero(self):
        a = 0
        self.m_gripper.SetTargetPosition(a)

    def jaw_to(self, jawwidth):
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state == 0):
            g_state = self.m_gripper.GetGripState()
            sleep(0.2)

    def mg_open(self):
        '''
        Main gripper open
        '''
        self.jaw_to(0.076)


    def mg_close(self):
        '''
        Main gripper open
        '''
        self.jaw_to(0)


    def mg_jaw_to(self, jawwidth):
        '''
        Main gripper jaws to "jawwidth"
        '''
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state == 0):
            g_state = self.m_gripper.GetGripState()
            sleep(0.2)


    def mg_get_jawwidth(self):
        '''
        Get current jawwidth of main gripper
        '''
        pass