import math
import numpy as np
# import robot_sim.robots.ur5_dual.ur5_dual_dh60 as two_ur5e
import robot_sim.robots.gofa5.gofa5_Ag145 as gofa5
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import manipulation.pick_place_planner as ppp
# import robot_sim.end_effectors.gripper.dh60.dh60 as hnd
import robot_sim.end_effectors.gripper.ag145.ag145 as hnd
import basis.robot_math as rm

if __name__ == '__main__':

    base = wd.World(cam_pos=[3, 0, 1.3], lookat_pos=[0, 0, .2])
    robot_s = gofa5.GOFA5()
    print(robot_s.is_collided())
    robot_s.gen_meshmodel().attach_to(base)
    original_grasp_info_list = gpa.load_pickle_file('xxx1', './', 'ag145_x.pickle')

    print(original_grasp_info_list)
    # base.run()

    workpiece_before = cm.CollisionModel("objects/xxx1.stl")
    # workpiece_after = cm.CollisionModel("objects/xxx1.stl")
    workpiece_after = workpiece_before.copy()
    workpiece_before.set_pos(pos=np.array([0.837, -0.05090143, 0.32737311]))
    workpiece_before.attach_to(base)

    workpiece_before.set_rgba(rgba=[1, 0, 0, 1])

    workpiece_after.set_pos(pos=np.array([0.837, -0.05090143, 0.32737311]))
    workpiece_after.attach_to(base)
    # base.run()
    obj_homo = workpiece_before.get_homomat()
    print(obj_homo)
    print('////')
    gripper_s = hnd.Ag145()
    # base.run()
    for i, item in enumerate(original_grasp_info_list):
        gri_homo = rm.homomat_from_posrot(item[3], item[4])
        print(gri_homo)
        print('//')
        hnd_homo = obj_homo.dot(gri_homo)
        fgr_homo = obj_homo.dot(rm.homomat_from_posrot(item[1], item[2]))
        gripper_s.fix_to(pos=hnd_homo[:3, 3], rotmat=hnd_homo[:3, :3])

        print(hnd_homo[:3, 3])
        print('//')
        # base.run()
        # gripper_s.gen_meshmodel(rgba=[0,1,0,1]).attach_to(base)
        try:
            robot_s.hnd.jaw_to(item[0])
            robot_s.gen_meshmodel().attach_to(base)
            gripper_s.jaw_to(item[0])
            gripper_s.gen_meshmodel().attach_to(base)
            jnts = robot_s.ik(component_name='arm',
                              tgt_pos=fgr_homo[:3, 3],
                              tgt_rotmat=fgr_homo[:3, :3],
                              seed_jnt_values=None,
                              max_niter=200,
                              tcp_jnt_id=None,
                              tcp_loc_pos=None,
                              tcp_loc_rotmat=None,
                              toggle_debug=False)
            print(jnts)
            robot_s.fk('arm', jnts)
            robot_s.gen_meshmodel(rgba=[0, 1, 0, 0.5]).attach_to(base)
            break
        except:
            print(i)
            pass
    base.run()
