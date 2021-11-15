import sys
sys.path.append('./')

import cv2
#import keyboard
import numpy as np
import open3d as o3d
#import pygame
from transforms3d.axangles import axangle2mat

import config_tf as config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils_tf import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils_tf import *
from vis import CVplot2D


def live_application(capture):
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  ############ output visualization ############
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  
  ret = False
  while not ret:
    ret, frame = capture.read()
  
  window_size = frame.shape[1]

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)

  view_control = viewer.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  extrinsic = cam_params.extrinsic.copy()
  extrinsic[0:3, 3] = 0
  cam_params.extrinsic = extrinsic
  cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    window_size // 2, window_size // 2
  )
  view_control.convert_from_pinhole_camera_parameters(cam_params)
  view_control.set_constant_z_far(1000)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()

  ############ input visualization ############
  #pygame.init()
  #display = pygame.display.set_mode((window_size, window_size))
  #pygame.display.set_caption('Minimal Hand - input')

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  #clock = pygame.time.Clock()
  model = ModelPipeline()

  while True:
    ret, frame_large = capture.read()
    if ret is None:
      continue
    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]

    input_img = frame_large.copy()
    h, w, c = input_img.shape
    frame_large = np.flip(frame_large, axis=1).copy()
    ratio = h / 128
    frame = imresize(frame_large, (128, 128))
    frame = frame[:, :, ::-1] # BGR2RGB

    uv, _, theta_mpii = model.process(frame)
    uv = uv.astype(float)
    theta_mano = mpii_to_mano(theta_mpii)

    v = hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v[:, 0] *= -1
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    viewer.poll_events()

    arr = np.asarray(viewer.capture_screen_float_buffer()) * 255
    arr = cv2.resize(arr, (w, h))
    render_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR).astype(np.uint8)
    render_img = cv2.resize(render_img, (320, 320))

    uv *= ratio
    uv[:, 0] = w - uv[:, 0]
    input_img = CVplot2D(input_img, uv[None, ...])
    input_img = cv2.resize(input_img, (320, 320))
    result = np.hstack((input_img, render_img.copy()))
    cv2.imshow('aa', result)
    if cv2.waitKey(1) == ord('q'):
      break


if __name__ == '__main__':
  live_application(cv2.VideoCapture(0))
