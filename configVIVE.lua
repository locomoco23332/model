
config = {
	--loaderFile=projectPath.."/seok_motion/hyunwoo_lowdof_T_boxfoot_parable.wrl",
	loaderFile='../../taesooLib/Resource/motion/Mixamo/passive_nomarker_man_T_nofingers.fbx.dat',
	motFile=projectPath.."/../../lafan1/lafan1_passive_nomarker_man_nofingers.mot2",
	skinScale = 100,
	rendering_step = 1/30,
	frameRate=30,
	trackerBones ={
		-- 가슴 센서
		{name='Spine',  trackerTransf=transf(quater(1, 0,0,0), vector3(0,0.1,0.1))},

		-- 발목에 센서를 달았다는 가정
		{name='LeftLeg',  trackerTransf=transf(quater(1, 0,0,0), vector3(0,-0.35,0.02))}, -- 무릎 관절에서 부터의 offset
		{name='RightLeg',   trackerTransf=transf(quater(1, 0,0,0), vector3(0,-0.35,0.02))},

		-- 팔목에 센서를 달았다는 가정
		{name='LeftForeArm',    trackerTransf=transf(quater(1, 0,0,0), vector3(0.2,0.05,0))},
		{name='RightForeArm',   trackerTransf=transf(quater(1, 0,0,0), vector3(-0.2,0.05,0))},

		-- 이마 앞에 센서를 달았다는 가정
		{name='Head',    trackerTransf=transf(quater(1, 0,0,0), vector3(0,0.13,0.13))},
	},
	trackerPosScale=1.0, -- used only with real-sensor data.
	useFilter=2, -- used only with real-sensor data. (this introduces additional delay. set nil to disable)

}

-- these three are the adjustable parameters
PAST_WINDOW=21 -- has to be an odd number
FUTURE_WINDOW=21
DELAY=10 -- so the future window becomes (iframe-DELAY, iframe-DELAY+FUTURE_WINDOW)
NUM_siMLPe_BLOCKS=48
