import math
import numpy as np
from common.realtime import sec_since_boot, DT_MDL
from common.numpy_fast import interp
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from selfdrive.controls.lib.drive_helpers import CONTROL_N, MPC_COST_LAT, LAT_MPC_N, CAR_ROTATION_RADIUS
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE , STEERING_CENTER
from selfdrive.config import Conversions as CV
import cereal.messaging as messaging
from cereal import log

STEERING_CENTER_calibration = []

LaneChangeState = log.LateralPlan.LaneChangeState
LaneChangeDirection = log.LateralPlan.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 30 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

DESIRES = {
  LaneChangeDirection.none: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.none,
  },
  LaneChangeDirection.left: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.laneChangeLeft,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.laneChangeLeft,
  },
  LaneChangeDirection.right: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.laneChangeRight,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.laneChangeRight,
  },
}


class LateralPlanner:
  def __init__(self, CP, use_lanelines=True, wide_camera=False):
    self.use_lanelines = use_lanelines
    self.LP = LanePlanner(wide_camera)

    self.last_cloudlog_t = 0
    self.steer_rate_cost = CP.steerRateCost

    self.solution_invalid_cnt = 0
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.lane_change_ll_prob = 1.0
    self.keep_pulse_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.LateralPlan.Desire.none

    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.path_xyz_stds = np.ones((TRAJECTORY_SIZE, 3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros(TRAJECTORY_SIZE)

    self.lat_mpc = LateralMpc()
    self.reset_mpc(np.zeros(6))

  def reset_mpc(self, x0=np.zeros(6)):
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def update(self, sm):
    v_ego = sm['carState'].vEgo
    active = sm['controlsState'].active
    measured_curvature = sm['controlsState'].curvature

    md = sm['modelV2']
    self.LP.parse_model(sm['modelV2'])
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE:
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      self.t_idxs = np.array(md.position.t)
      self.plan_yaw = list(md.orientation.z)
    if len(md.position.xStd) == TRAJECTORY_SIZE:
      self.path_xyz_stds = np.column_stack([md.position.xStd, md.position.yStd, md.position.zStd])

    STEER_CTRL_Y = sm['carState'].steeringAngleDeg
    path_y = self.path_xyz[:,1]
    max_yp = 0
    for yp in path_y:
      max_yp = yp if abs(yp) > abs(max_yp) else max_yp
    STEERING_CENTER_calibration_max = 300 #3秒
    if abs(max_yp) / 2.5 < 0.1 and v_ego > 20/3.6 and abs(STEER_CTRL_Y) < 8:
      STEERING_CENTER_calibration.append(STEER_CTRL_Y)
      if len(STEERING_CENTER_calibration) > STEERING_CENTER_calibration_max:
        STEERING_CENTER_calibration.pop(0)
    if len(STEERING_CENTER_calibration) > 0:
      value_STEERING_CENTER_calibration = sum(STEERING_CENTER_calibration) / len(STEERING_CENTER_calibration)
    else:
      value_STEERING_CENTER_calibration = 0
    handle_center = STEERING_CENTER
    if len(STEERING_CENTER_calibration) >= STEERING_CENTER_calibration_max:
      handle_center = value_STEERING_CENTER_calibration #動的に求めたハンドルセンターを使う。
      with open('./handle_center_info.txt','w') as fp:
        fp.write('%0.2f' % (value_STEERING_CENTER_calibration) )
    with open('./debug_out_y','w') as fp:
      path_y_sum = -sum(path_y)
    #  #fp.write('{0}\n'.format(['%0.2f' % i for i in self.path_xyz[:,1]]))
      fp.write('calibration:%0.2f/%d ; max:%0.2f ; sum:%0.2f ; avg:%0.2f' % (value_STEERING_CENTER_calibration,len(STEERING_CENTER_calibration),-max_yp , path_y_sum, path_y_sum / len(path_y)) )
    STEER_CTRL_Y -= handle_center #STEER_CTRL_Yにhandle_centerを込みにする。
    ypf = STEER_CTRL_Y
    if abs(STEER_CTRL_Y) < abs(max_yp) / 2.5:
      STEER_CTRL_Y = (-max_yp / 2.5)

    if False:
      ssa = ""
      ssao = ""
      ssas = ""
      if ypf > 0:
        for vml in range(int(min(ypf,30))):
          ssa+= "|"
      if ypf > 0 and int(min(STEER_CTRL_Y - ypf,30 - len(ssa))) > 0:
        for vml in range(int(min(STEER_CTRL_Y - ypf,30 - len(ssa)))):
          ssao+= "<"
      elif ypf < 0 and (STEER_CTRL_Y) > 0 and int(min((STEER_CTRL_Y),30 - len(ssa))) > 0:
        for vml in range(int(min((STEER_CTRL_Y),30 - len(ssa)))):
          ssao+= "<"
      if 30 - len(ssa) - len(ssao) > 0:
        for vml in range(int(30 - len(ssa) - len(ssao))):
          ssas+= " "
      mssa = ""
      mssao = ""
      mssas = ""
      if ypf < 0:
        for vml in range(int(min(-ypf,30))):
          mssa+= "|"
      if ypf < 0 and int(min(-(STEER_CTRL_Y - ypf),30 - len(mssa))) > 0:
        for vml in range(int(min(-(STEER_CTRL_Y - ypf),30 - len(mssa)))):
          mssao+= ">"
      elif ypf > 0 and (STEER_CTRL_Y) < 0 and int(min(-(STEER_CTRL_Y),30 - len(mssa))) > 0:
        for vml in range(int(min(-(STEER_CTRL_Y),30 - len(mssa)))):
          mssao+= ">"
      if 30 - len(mssa) - len(mssao) > 0:
        for vml in range(int(30 - len(mssa) - len(mssao))):
          mssas+= " "
      with open('./debug_out_1','w') as fp:
        #fp.write('strAng:%0.1f->%0.1f[deg] , speed:%0.1f[km/h]' % (ypf , STEER_CTRL_Y - ypf, v_ego * 3.6))
        #fp.write('steerAngY:%0.1f[deg] , speed:%0.1f[km/h]' % (STEER_CTRL_Y, v_ego * 3.6))
        #fp.write('steerAng:%0.1f[deg] , speed:%0.1f[km/h]' % (STEER_CTRL_Y + handle_center, v_ego * 3.6)) #ハンドルセンターなしの素のSTEER_CTRL_Yを表示
        fp.write('strAng:%5.1f(%+5.1f[deg])%s%s%s^%s%s%s' % (ypf , STEER_CTRL_Y - ypf, ssas,ssao,ssa,mssa,mssao,mssas))
      

    if sm['carState'].leftBlinker == True:
      STEER_CTRL_Y = 90
    if sm['carState'].rightBlinker == True:
      STEER_CTRL_Y = -90

    # Lane change logic
    one_blinker = sm['carState'].leftBlinker != sm['carState'].rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    if (not active) or (self.lane_change_timer > LANE_CHANGE_TIME_MAX):
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      # LaneChangeState.off
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_ll_prob = 1.0

      # LaneChangeState.preLaneChange
      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Set lane change direction
        if sm['carState'].leftBlinker:
          self.lane_change_direction = LaneChangeDirection.left
        elif sm['carState'].rightBlinker:
          self.lane_change_direction = LaneChangeDirection.right
        else:  # If there are no blinkers we will go back to LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none

        torque_applied = sm['carState'].steeringPressed and \
                         ((sm['carState'].steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (sm['carState'].steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((sm['carState'].leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (sm['carState'].rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting

      # LaneChangeState.laneChangeStarting
      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        # fade out over .5s
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2 * DT_MDL, 0.0)

        # 98% certainty
        lane_change_prob = self.LP.l_lane_change_prob + self.LP.r_lane_change_prob
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
          self.lane_change_state = LaneChangeState.laneChangeFinishing

      # LaneChangeState.laneChangeFinishing
      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        # fade in laneline over 1s
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)
        if self.lane_change_ll_prob > 0.99:
          self.lane_change_direction = LaneChangeDirection.none
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
          else:
            self.lane_change_state = LaneChangeState.off

    if self.lane_change_state in [LaneChangeState.off, LaneChangeState.preLaneChange]:
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker

    self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

    # Send keep pulse once per second during LaneChangeStart.preLaneChange
    if self.lane_change_state in [LaneChangeState.off, LaneChangeState.laneChangeStarting]:
      self.keep_pulse_timer = 0.0
    elif self.lane_change_state == LaneChangeState.preLaneChange:
      self.keep_pulse_timer += DT_MDL
      if self.keep_pulse_timer > 1.0:
        self.keep_pulse_timer = 0.0
      elif self.desire in [log.LateralPlan.Desire.keepLeft, log.LateralPlan.Desire.keepRight]:
        self.desire = log.LateralPlan.Desire.none

    # Turn off lanes during lane change
    if self.desire == log.LateralPlan.Desire.laneChangeRight or self.desire == log.LateralPlan.Desire.laneChangeLeft:
      self.LP.lll_prob *= self.lane_change_ll_prob
      self.LP.rll_prob *= self.lane_change_ll_prob
    if self.use_lanelines:
      #d_path_xyz = self.LP.get_d_path(v_ego, self.t_idxs, self.path_xyz)
      d_path_xyz = self.LP.get_d_path(STEER_CTRL_Y , v_ego, self.t_idxs, self.path_xyz)
      self.lat_mpc.set_weights(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, self.steer_rate_cost)
    else:
      d_path_xyz = self.path_xyz
      dcm = self.LP.calc_dcm(STEER_CTRL_Y, v_ego,2.5,-1,-1) #2.5はレーンを消すダミー,-1,-1はカメラオフセット反映に必要
      d_path_xyz[:,1] -= dcm #CAMERA_OFFSETが反映されている。->実はcalc_dcmの中で無視している。無い方が走りが良い？
      path_cost = np.clip(abs(self.path_xyz[0, 1] / self.path_xyz_stds[0, 1]), 0.5, 1.5) * MPC_COST_LAT.PATH
      # Heading cost is useful at low speed, otherwise end of plan can be off-heading
      heading_cost = interp(v_ego, [5.0, 10.0], [MPC_COST_LAT.HEADING, 0.0])
      self.lat_mpc.set_weights(path_cost, heading_cost, self.steer_rate_cost)
    y_pts = np.interp(v_ego * self.t_idxs[:LAT_MPC_N + 1], np.linalg.norm(d_path_xyz, axis=1), d_path_xyz[:, 1])
    heading_pts = np.interp(v_ego * self.t_idxs[:LAT_MPC_N + 1], np.linalg.norm(self.path_xyz, axis=1), self.plan_yaw)
    self.y_pts = y_pts

    assert len(y_pts) == LAT_MPC_N + 1
    assert len(heading_pts) == LAT_MPC_N + 1
    self.x0[4] = v_ego
    self.lat_mpc.run(self.x0,
                     v_ego,
                     CAR_ROTATION_RADIUS,
                     y_pts,
                     heading_pts)
    # init state for next
    self.x0[3] = interp(DT_MDL, self.t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3])

    #  Check for infeasible MPC solution
    mpc_nans = any(math.isnan(x) for x in self.lat_mpc.x_sol[:, 3])
    t = sec_since_boot()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      self.reset_mpc()
      self.x0[3] = measured_curvature
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 20000. or mpc_nans:
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState', 'modelV2'])
    plan_send.lateralPlan.laneWidth = float(self.LP.lane_width)
    plan_send.lateralPlan.dPathPoints = [float(x) for x in self.y_pts]
    plan_send.lateralPlan.psis = [float(x) for x in self.lat_mpc.x_sol[0:CONTROL_N, 2]]
    plan_send.lateralPlan.curvatures = [float(x) for x in self.lat_mpc.x_sol[0:CONTROL_N, 3]]
    plan_send.lateralPlan.curvatureRates = [float(x) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [0.0]
    plan_send.lateralPlan.lProb = float(self.LP.lll_prob)
    plan_send.lateralPlan.rProb = float(self.LP.rll_prob)
    plan_send.lateralPlan.dProb = float(self.LP.d_prob)

    plan_send.lateralPlan.mpcSolutionValid = bool(plan_solution_valid)

    plan_send.lateralPlan.desire = self.desire
    plan_send.lateralPlan.useLaneLines = self.use_lanelines
    plan_send.lateralPlan.laneChangeState = self.lane_change_state
    plan_send.lateralPlan.laneChangeDirection = self.lane_change_direction

    pm.send('lateralPlan', plan_send)
