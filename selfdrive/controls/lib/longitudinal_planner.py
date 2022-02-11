#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import interp

import cereal.messaging as messaging
from common.realtime import DT_MDL
from selfdrive.modeld.constants import T_IDXS
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N
from selfdrive.swaglog import cloudlog

from selfdrive.car.toyota.values import TSS2_CAR
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE , STEERING_CENTER
from common.params import Params
PARAMS = Params()
CVS_FRAME = 0

def calc_limit_vc(X1,X2,X3 , Y1,Y2,Y3):
  Z1 = (X2-X1)/(Y1-Y2) - (X3-X2)/(Y2-Y3)
  Z2 = (X3-X2)/(Y2-Y3) - (X1-X3)/(Y3-Y1)
  A = (X2-X1)*(X1*X2 - X2*X3) - (X1-X3)*(X2*X3 - X3*X1)
  A /= Z1*(X2-X1) - Z2*(X1-X3)
  B = ((X1*X2 - X2*X3) - A*Z1) / (X1-X3)
  C = Y1 - A / (X1 - B)
  return (A,B,C)

LIMIT_VC_A ,LIMIT_VC_B ,LIMIT_VC_C  = calc_limit_vc(8.7,11.6,27.0 , 86-4      ,60-4      ,47-4      )
#LIMIT_VC_AH,LIMIT_VC_BH,LIMIT_VC_CH = calc_limit_vc(8.7,11.6,23.0 , 96-4+3+1+4,72-4+4+3+3,60-4+5+6+2)
LIMIT_VC_AH,LIMIT_VC_BH,LIMIT_VC_CH = calc_limit_vc(8.7,13.0,25.0 , 112,93,81)

OP_ENABLE_PREV = False
OP_ENABLE_v_cruise_kph = 0
OP_ENABLE_gas_speed = 0

LON_MPC_STEP = 0.2  # first step is 0.2s
AWARENESS_DECEL = -0.2  # car smoothly decel at .2m/s^2 when user is distracted
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_VALS = [1.2, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 15., 25., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class Planner:
  def __init__(self, CP, init_v=0.0, init_a=0.0):
    self.CP = CP
    self.mpc = LongitudinalMpc()

    self.fcw = False

    self.v_desired = init_v
    self.a_desired = init_a
    self.alpha = np.exp(-DT_MDL / 2.0)

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    v_ego = sm['carState'].vEgo
    a_ego = sm['carState'].aEgo

    v_cruise_kph = sm['controlsState'].vCruise
    if CP.carFingerprint not in TSS2_CAR:
      v_cruise_kph = (55 - (55 - (v_cruise_kph+4)) * 2 - 4) if v_cruise_kph < (55 - 4) else v_cruise_kph
      v_cruise_kph = (110 + ((v_cruise_kph+6) - 110) * 3 - 6) if v_cruise_kph > (110 - 6) else v_cruise_kph
      if CVS_FRAME % 5 == 3 and CVS_FRAME < 30:
        with open('./tss_type_info.txt','w') as fp:
          fp.write('%d' % (1))
    else:
      if CVS_FRAME % 5 == 3 and CVS_FRAME < 30:
        with open('./tss_type_info.txt','w') as fp:
          fp.write('%d' % (2))
    global OP_ENABLE_PREV
    global OP_ENABLE_v_cruise_kph
    global OP_ENABLE_gas_speed
    #if OP_ENABLE_PREV == False and sm['controlsState'].longControlState != LongCtrlState.off and v_ego > 3/3.6: # and sm['carState'].gasPressed:
       #速度が時速３km以上なら、無条件にエクストラエンゲージするテスト 
    if OP_ENABLE_PREV == False and sm['controlsState'].longControlState != LongCtrlState.off and sm['carState'].gasPressed:
      #アクセル踏みながらのOP有効化の瞬間
      OP_ENABLE_v_cruise_kph = v_cruise_kph
      OP_ENABLE_gas_speed = v_ego
    if sm['controlsState'].longControlState != LongCtrlState.off:
      OP_ENABLE_PREV = True
      if sm['carState'].gasPressed:
        OP_ENABLE_gas_speed = v_ego
    else:
      OP_ENABLE_PREV = False
      OP_ENABLE_v_cruise_kph = 0
    if OP_ENABLE_v_cruise_kph != v_cruise_kph: #レバー操作したらエンゲージ初期クルーズ速度解除
      OP_ENABLE_v_cruise_kph = 0
    if OP_ENABLE_v_cruise_kph != 0:
      v_cruise_kph = OP_ENABLE_gas_speed*3.6 #エンゲージ初期クルーズ速度を優先して使う
    handle_center = STEERING_CENTER
    steerAng = sm['carState'].steeringAngleDeg - handle_center
    orgSteerAng = steerAng
    limit_vc = V_CRUISE_MAX
    limit_vc_h = V_CRUISE_MAX
    md = sm['modelV2']
    global CVS_FRAME
    ml_csv = ""
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE and PARAMS.get_bool("IsMetric"):
      path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      path_y = path_xyz[:,1]
      max_yp = 0
      for yp in path_y:
        max_yp = yp if abs(yp) > abs(max_yp) else max_yp
        if abs(steerAng) < abs(max_yp) / 2.5:
          steerAng = (-max_yp / 2.5)
      limit_vc = V_CRUISE_MAX if abs(steerAng) <= LIMIT_VC_B else LIMIT_VC_A / (abs(steerAng) - LIMIT_VC_B) + LIMIT_VC_C
      limit_vc_h = V_CRUISE_MAX if abs(steerAng) <= LIMIT_VC_BH else LIMIT_VC_AH / (abs(steerAng) - LIMIT_VC_BH) + LIMIT_VC_CH
      if CVS_FRAME % 10 == 0 and v_ego * 3.6 > 20: # over 20km/h
        ml_csv = '%0.2f,' % v_cruise_kph
        for i in path_y:
          ml_csv += '%0.2f,' % i
    v_cruise_kph_org = v_cruise_kph
    limit_vc_th = 95-5 #85-5 #80-4
    limit_vc_tl = 65-4 #70-4
    if v_cruise_kph_org > limit_vc_th:
      limit_vc = limit_vc_h
    elif v_cruise_kph_org >= limit_vc_tl:
      limit_vc = (limit_vc * ((limit_vc_th)-v_cruise_kph_org) + limit_vc_h * (v_cruise_kph_org - (limit_vc_tl))) / (limit_vc_th - limit_vc_tl)
    v_cruise_kph = limit_vc if limit_vc < v_cruise_kph else v_cruise_kph
    if CVS_FRAME % 5 == 2:
      with open('./limit_vc_info.txt','w') as fp:
        fp.write('%d' % (limit_vc))
    if CVS_FRAME % 5 == 1:
      with open('./steer_ang_info.txt','w') as fp:
        fp.write('%f' % (steerAng))
    if CVS_FRAME % 5 == 0:
      with open('./cruise_info.txt','w') as fp:
        #fp.write('%d/%d' % (v_cruise_kph_org , (limit_vc if limit_vc < V_CRUISE_MAX else V_CRUISE_MAX)))
        if v_cruise_kph == limit_vc:
          fp.write('%d.' % (v_cruise_kph))
        else:
          fp.write('%d' % (v_cruise_kph_org))
    #if CVS_FRAME % 10 == 0 and limit_vc < V_CRUISE_MAX and v_ego * 3.6 > 20: # over 20km/h
    #  with open('./ml_data.csv','a') as fp:
    #    fp.write('%s%0.2f\n' % (ml_csv , limit_vc))
    CVS_FRAME += 1

    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    long_control_state = sm['controlsState'].longControlState
    force_slow_decel = sm['controlsState'].forceDecel

    prev_accel_constraint = True
    if long_control_state == LongCtrlState.off or sm['carState'].gasPressed:
      self.v_desired = v_ego
      self.a_desired = a_ego
      # Smoothly changing between accel trajectory is only relevant when OP is driving
      prev_accel_constraint = False

    # Prevent divergence, smooth in current v_ego
    self.v_desired = self.alpha * self.v_desired + (1 - self.alpha) * v_ego
    self.v_desired = max(0.0, self.v_desired)

    if False:
      msv_desired = max(0,self.v_desired * 3.6)
      msc = "A:%5.1fkm/h" % (v_cruise_kph_org)
      if int(min(v_cruise_kph_org,V_CRUISE_MAX) / 2) - len(msc) > 0:
        for vm in range(int(min(v_cruise_kph_org,V_CRUISE_MAX) / 2) - len(msc)):
          msc += "#"
      msl = "L:%5.1fkm/h" % (limit_vc)
      if int(min(limit_vc,V_CRUISE_MAX) / 2) - len(msl) > 0:
        for vml in range(int(min(limit_vc,V_CRUISE_MAX) / 2) - len(msl)):
          msl += "<"
      v_ego_2 = max(0,v_ego * 3.6)
      msv = "V:%5.1fkm/h" % (v_ego_2)
      if msv_desired <= v_ego_2:
        if int(min(msv_desired,V_CRUISE_MAX) / 2) - len(msv) > 0:
          for vml in range(int(min(msv_desired,V_CRUISE_MAX) / 2) - len(msv)):
            msv += "|"
        if int(min(v_ego_2,V_CRUISE_MAX) / 2) - len(msv) > 0:
          for vml in range(int(min(v_ego_2,V_CRUISE_MAX) / 2) - len(msv)):
            msv += "<"
      else:
        if int(min(v_ego_2,V_CRUISE_MAX) / 2) - len(msv) > 0:
          for vml in range(int(min(v_ego_2,V_CRUISE_MAX) / 2) - len(msv)):
            msv += "|"
        if int(min(msv_desired,V_CRUISE_MAX) / 2) - len(msv) > 0:
          for vml in range(int(min(msv_desired,V_CRUISE_MAX) / 2) - len(msv)):
            msv += ">"
      msv += "%+.1fkm/h" % (msv_desired-v_ego_2)
      with open('./debug_out_v','w') as fp:
        #fp.write('[%i],vc:%.1f(%.1f) , v:%.2f , vd:%.2f[km/h] ; ah:%.2f bh:%.2f ch:%.2f' % (prev_accel_constraint , v_cruise_kph_org , limit_vc , v_ego * 3.6 , self.v_desired* 3.6 , LIMIT_VC_AH,LIMIT_VC_BH,LIMIT_VC_CH) )
        #fp.write('[%i],vc:%.1f(%.1f) , v:%.2f , vd:%.2f[km/h] ; a:%.2f , ad:%.2f[m/ss]' % (prev_accel_constraint , v_cruise_kph , limit_vc , v_ego * 3.6 , self.v_desired* 3.6 , a_ego , self.a_desired) )
        fp.write('ah:%.2f bh:%.2f ch:%.2f\n' % (LIMIT_VC_AH,LIMIT_VC_BH,LIMIT_VC_CH) )
        #fp.write('op:[%d] vk:%.2f gs:%.2fkm/h\n' % (OP_ENABLE_PREV,OP_ENABLE_v_cruise_kph,OP_ENABLE_gas_speed*3.6) )
        fp.write("%s\n%s\n%s" % (msc ,msl ,msv))

    accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]
    #accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    accel_limits_turns = limit_accel_in_turns(v_ego, orgSteerAng, accel_limits, self.CP)
    if force_slow_decel:
      # if required so, force a smooth deceleration
      accel_limits_turns[1] = min(accel_limits_turns[1], AWARENESS_DECEL)
      accel_limits_turns[0] = min(accel_limits_turns[0], accel_limits_turns[1])
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.05)
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired, self.a_desired)
    self.mpc.update(sm['carState'], sm['radarState'], v_cruise, prev_accel_constraint=prev_accel_constraint)
    self.v_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 5
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired = self.v_desired + DT_MDL * (self.a_desired + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']

    longitudinalPlan.speeds = [float(x) for x in self.v_desired_trajectory]
    longitudinalPlan.accels = [float(x) for x in self.a_desired_trajectory]
    longitudinalPlan.jerks = [float(x) for x in self.j_desired_trajectory]

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    pm.send('longitudinalPlan', plan_send)
