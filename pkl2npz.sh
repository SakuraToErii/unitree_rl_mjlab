source ./.venv/bin/activate
python scripts/gmr_to_npz_inter.py \
  --input_file "/home/eai/Downloads/MotionCode_1_take6_Skeleton0_TienKung3_RobotReady_120FPS - 20260812.pkl" \
  --input_fps 120 \
  --output_fps 100 \
  --frame_range 5 9999999999 \
  --output_name 1-1_paddingv2\
  --output_dir ./datasets \
  --start_frames 240 \
  --end_frames 240 \
  --hold_pos 360 \
  --joint_limit_factor 1.0 \
  --correct_root_pose_coupled