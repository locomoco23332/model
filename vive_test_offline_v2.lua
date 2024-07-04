require("config")
package.projectPath='../Samples/classification/'
package.path=package.path..";../Samples/classification/lua/?.lua" --;"..package.path
require("common")
require("module")
fc=require('retargetting/module/footSkateCleanup')
require('subRoutines/Timeline')
require("RigidBodyWin/subRoutines/Constraints")
-- require("gym_seok/play_ik")
require("RigidBodyWin/subRoutines/CollisionChecker")
FBXloader=require('FBXloader')

scriptPath=util.getScriptPath()
projectPath=os.parentPath(scriptPath)
package.path=package.path..';'..scriptPath..'/?.lua'
require("extractFeatures")
require("Model1") -- AE10 model (unfiltered referenceCoord).
require("Model2") -- TF model (unfiltered referenceCoord).
require("Model3") -- siMLPe model (filtered referenceCoord).
require("Model4") -- siMLPe model (unfiltered referenceCoord).
require("Model5") -- GAVAE
require("Model6") -- TGAVAE
require("Model7") -- CVAE
require("Model8")
require("Model9")
require("Model10")
require("ikHelper")

config.numTracker = #config.trackerBones

function handleRendererEvent()
	return 0
end
function ctor()
	--dbg.startTrace2()

	mEventReceiver = EVR()
    local osm =RE.ogreSceneManager()
	if osm then
		osm:setFogNone()
	end
	--mLoader=MainLib.VRMLloader ("../Resource/motion/locomotion_hyunwoo/hyunwoo_lowdof_T_boxfoot.wrl")
	mFbxLoader=FBXloader (config.loaderFile) mLoader=mFbxLoader.loader
	mLoader2=mLoader -- finger 아직 안씀.
	--mLoader2= MainLib.VRMLloader ("../seok_motion/hyunwoo_lowdof_T_boxfoot_parable.wrl")
	if config.useFilter then
		local filterSize=config.useFilter*2+1 -- has to be an odd number

		local initialPose=vectorn()
		mLoader2:updateInitialBone()
		mLoader2:getPoseDOF(initialPose)
		mFilter=OnlineFilter(mLoader2, initialPose, filterSize, true)
	end

	for i, bone in ipairs(config.trackerBones) do
		local bone2=mLoader2:getBoneByName(bone.name)
		bone.idx=bone2:treeIndex()
		bone.startT=mLoader2.dofInfo:startT(bone2:treeIndex()) -- posedof의 스타트 인덱스. 
		bone.idx1=mLoader:getBoneByName(bone.name):treeIndex()
	end

	--DOFcontainer = MotionDOFcontainer(mLoader2.dofInfo , "../seok_motion/stitchMotion_mixamo.dof")
	--mMotionDOF = DOFcontainer.mot
	mSkin2 = RE.createFBXskin(mFbxLoader,false)
	mSkin2:setScale(config.skinScale,config.skinScale,config.skinScale)

	--vive class initilze 하는 부분--
    --vive = CHtc_Vive_Tracker()
    --vive:InitializeVR(true) 
	local viveMocap = loadViveRaw()
	mocap = matrixn()
	for i=0, 6-1 do
		local tmpTrackerPos = viveMocap.pos:sub(0,0, 3*i,3*i+3) -- 0~2, 3~5
		local tmpTrackerOri = viveMocap.ori:sub(0,0, 4*i,4*i+4) -- 0~3, 4~7
		if i == 0 then
			mocap = tmpTrackerPos..tmpTrackerOri
		else 
			mocap = mocap..tmpTrackerPos..tmpTrackerOri
		end
	end

	prev_pose = vectorn(mLoader.dofInfo:numDOF())
	tracker_history = CT.zeros(PAST_WINDOW,config.numTracker*7)
	iframe = 0 
	RE.viewpoint():update()

	this:addButton('init AE10 (model1)')
	this:addButton('init TF (model2)')
	this:addButton('init siMLPe (model3)')
	this:addButton('init siMLPe (model4)')
	this:addButton('init GAVAE (model5)')
	this:addButton('init TGAVAE (model6)')
	this:addButton('init CVAE (model7)')
	this:addButton('init CNN (model8)')
	this:addButton('init DanceVAE (model9)')
	this:addButton('init Belfusion (model10)')
	this:addCheckButton('solve IK', false)
	this:addCheckButton('solve swivel-angle', false)
	this:updateLayout()
	calibrate = false
	mTimeline=Timeline('timeline', 10000000, config.rendering_step)

	get_delta = true
	use_siMLPe4()
	createIKsolver()
	
end

function use_AE10()
	model=Model1(config )
	model:createPyModule()
end
function use_TF()
	model=Model2(config )
	model:createPyModule()
end
function use_siMLPe3()
	model=Model3(config )
	model:createPyModule()
end
function use_siMLPe4()
	model=Model4(config)
	model:createPyModule()
end
function use_GAVAE()
      model=Model5(config)
      model:createPyModule()
end
function use_TGAVAE()
	model=Model6(config)
	model:createPyModule()
end
function use_CVAE()
	 model=Model7(config)
	model:createPyModule()
end
function use_CNN()
	model=Model8(config)
	model:createPyModule()
end
function use_DanceVAE()
	model=Model9(config)
	model:createPyModule()
end
function use_Belfusion()
	model=Model10(config)
	model:createPyModule()
end
function onCallback(w, userData)
	if w:id()=='init AE10 (model1)' then
		print('init model1')
		use_AE10()
	elseif w:id()=='init TF (model2)' then
		print('init model2')
		use_TF()
	elseif w:id()=='init siMLPe (model3)' then
		print('init model3')
		use_siMLPe3()
	elseif w:id()=='init siMLPe (model4)' then
		print('init model4')
		use_siMLPe4()
	elseif w:id()=='init GAVAE (model5)' then
	        print('init model5')
	        use_GAVAE()
	elseif w:id()=='init TGAVAE (model6)' then
		    print('init model6')
		    use_TGAVAE()
	elseif w:id()=='init CVAE (model7)' then
		    print('init model7')
		    use_CVAE()
	elseif w:id()=='init CNN (model8)' then
		     print('init model8')
		     use_CNN()
	elseif w:id()=='init DanceVAE (model9)' then
		      print('init model9')
		      use_DanceVAE()
	elseif w:id()=='init Belfusion (model10)' then
		     print('init model10')
		     use_Belfusion()

	end

end

--calibrate 할때 delta 구하는 함수--
function calcDelta(mLoader,tracker_data)
	local refDelta = matrixn(6,7)
	for i=0,5 do 
		refDelta:row(i):setTransf(0, mLoader:bone(config.trackerBones[i+1].idx1):getFrame():inverse()*tracker_data:toTransf(7*i))
	end
	return refDelta
end

if EventReceiver then 
	EVR = LUAclass(EventReceiver)
	function EVR:__init(graph)
		self.currFrame = 0
		self.cameraInfo = {}
	end 
end	

function dtor()
end

--tracker index로 하나의 pos,ori 데이터 불러오는 함수--
function getTrackerDataByIndex(device_id)
	local tracker = vectorn(7)
	tracker:assign(vive:getTrackerDataByIndex(device_id))
	return tracker
end

function EVR:onFrameChanged(win,iframe)
	updateFrame(iframe)
	dbg.delayedDrawTick()
end
function updateFrame(iframe)
	if iframe>=mocap:rows() then return end
    --local tracker_data = vive:GetOnlyTrackerData()
	local tracker_data = mocap:row(iframe)
	--y축 임의의 값으로 올리기--
	for i=0,5 do 
		tracker_data:set(7*i+1,tracker_data(7*i+1)+2)
	end
	
	--button 누를때마다 델타값 계산해서 calibration 진행--
	if get_delta then 
		local coord = quater()
		local leftAnkle = tracker_data:toTransf(7)
		local rightAnkle = tracker_data:toTransf(14)

		coord:setAxisRotation(vector3(0,1,0),vector3(1,0,0),leftAnkle.translation-rightAnkle.translation)
		local pose = vectorn(mLoader.dofInfo:numDOF())
		pose:setAllValue(0)
		pose:setVec3(0,tracker_data:toVector3(0))
		pose:setQuater(3,coord)
		mLoader:setPoseDOF(pose)
		refDelta = calcDelta(mLoader,tracker_data)
		get_delta = false
		calibrate = true
		Ydelta = 1 - tracker_data:toVector3(0).y
		iframe = 0
		t_frame = iframe 
	end

	--calibration 하는 과정--
	calibrated_tracker = vectorn(42)
	if calibrate == true then 
		for i=0,5 do 
			--dbg.delayedDraw('Axes',10, tracker_data:toTransf(7*i),'o'..i,100)
			calibrated_tracker:setTransf(7*i,calibrate_tracker(tracker_data:toTransf(7*i),refDelta:row(i):toTransf()))
		end
	else
		calibrated_tracker = tracker_data
	end

	--if iframe-t_frame > 11 then 
	if tracker_history:row(0):length()>0 
		and iframe-t_frame >= 40  and tracker_history:rows()>20 then 

		--Draw(tracker_history:row(11))
		Draw(tracker_history:row(tracker_history:rows()-1))
		--local irow=this:findWidget("predictionFrame"):sliderValue()+DELAY
		local irow=0 -- show the most robust (and the most outdated) pose
		local pose=model:getPose(irow) -- uses tracker_history

		if this:findWidget('solve IK'):checkButtonValue() then
			local td=tracker_history:row(tracker_history:rows()-1+irow-DELAY)
			solveIK(pose, td, this:findWidget('solve swivel-angle'):checkButtonValue())
		end

		if mFilter and pose then
			mFilter:setCurrPose(pose)
			local renderPoseFiltered=mFilter:getFiltered()
			mSkin2:setPoseDOF(renderPoseFiltered)
			mLoader2:setPoseDOF(renderPoseFiltered)
		elseif pose then
			mSkin2:setPoseDOF(pose)
			mLoader2:setPoseDOF(pose)
		end
	end

	--tracker history 에 칼리브레이션된 트레커 데이터 push--
	pushBack(tracker_history,calibrated_tracker)
end

function getMidCoord(a,b)
	assert(lunaType(a)=='transf','a is not trnasf')
	assert(lunaType(b)=='transf','b is not transf')
	local output = transf()
	output:interpolate(0.5,a,b)
	return output 
end

function Draw(tracker_data)
	local tracker = matrixn(6,7)
	for i=0,tracker:rows()-1 do 
		tracker:row(i):assign(tracker_data:range(7*i,7+7*i))
		local coord = tracker:row(i):toTransf()
		dbg.delayedDraw('Axes',DELAY, coord,''..i,100)
		--dbg.namedDraw('Axes',coord,''..i,100)
	end
end


function calibrate_tracker(tracker_data,delta)
	tracker_data.rotation:assign(tracker_data.rotation*delta.rotation:inverse())
	tracker_data.translation.y = tracker_data.translation.y + Ydelta
	tracker_data.translation:scale(config.trackerPosScale)
    return tracker_data
end

function motionMatching2Parable(pose)
	assert(pose:size() == 33 ,"원본 pose 사이즈 안맞음")
	local output_pose = vectorn(89)
	output_pose:range(0,26):assign(pose:range(0,26))
	local left_hand = vectorn(28)
	left_hand:setAllValue(0)
	output_pose:range(26,54):assign(left_hand)
	output_pose:range(54,58):assign(pose:range(26,30))
	local right_hand = vectorn(28)
	right_hand:setAllValue(0)
	output_pose:range(58,86):assign(right_hand)
	output_pose:range(86,89):assign(pose:range(30,33))
	return output_pose
end


function pushBack(matrix,data)
	for i=0,matrix:rows()-2 do
		matrix:row(i):assign(matrix:row(i+1))
	end
	matrix:row(matrix:rows()-1):assign(data)
end



elapsedTime = 0
function frameMove(fElapsedTime)
end

-- kang ----------------------------------------------------------------------
function loadViveRaw()
	--local chosenFile=Fltk.chooseFile("Choose a extracted feature file", './', "*.dat", false)
	local chosenFile = projectPath.."/vive/viveCapture2023-05-09_133114.dat" --복싱
	--local chosenFile = "./parable/vive/viveCapture2023-05-30_170718.dat"
	--local chosenFile = "./parable/vive/viveCapture2023-05-30_171733.dat"
	--local chosenFile = "./parable/vive/viveCapture2023-05-30_173213.dat"
	--local chosenFile = "./parable/vive/viveCapture2023-05-30_173319.dat"
	--local chosenFile = "./parable/vive/viveCapture2023-05-30_174111.dat"
	local mocap = util.loadTable(chosenFile)
	
	-- remove redundant frames
	--local startFrame = math.floor(mocap.ori:rows() - mocap.ori:rows()/4.5) -- ratio that you will remove 
	local startFrame = 0
	--local startFrame = 8000
	local remainOri = mocap.ori:sub(startFrame,0, 0,0) -- row: mass~end  col: 0~end
	local remainPos = mocap.pos:sub(startFrame,0, 0,0)
	local vive = {}
	local numRow = remainOri:rows()
	local numCol = remainOri:cols()
	vive.ori = matrixn(numRow, numCol)
	vive.pos = matrixn(numRow, numCol)
	--vive.ori:resampleFrameSkip(remainOri, 8) -- frame rate resample roughly 250>30FPS
	--vive.pos:resampleFrameSkip(remainPos, 8)
	vive.ori = remainOri
	vive.pos = remainPos

	-- Translation for convenience
	-- cross (Hips,Neck), (LeftArm,RightArm)  -> forward dir
	---- compute angle diff forward dir
	--[[
	local loaderHipPos = mLoader:getBoneByName(config.bonesNames.hips.name):getFrame().translation
	local loaderNeckPos = mLoader:getBoneByName(config.bonesNames.neck.name):getFrame().translation
	local loaderLeftArmPos = mLoader:getBoneByName(config.bonesNames.left_elbow.name):getFrame().translation
	local loaderRightArmPos = mLoader:getBoneByName(config.bonesNames.right_elbow.name):getFrame().translation
	local loaderUpVec = loaderNeckPos - loaderHipPos
	loaderUpVec:normalize()
	local loaderRightVec = loaderRightArmPos - loaderLeftArmPos
	loaderRightVec:normalize()
	local viveInitialRow = vive.pos:row(0)
	-- !!!!! 힙,왼다리,오른다리,왼팔,오른팔,목 정렬 됐다고 가정.
	local viveHipPos = viveInitialRow:toVector3(0)
	local viveNeckPos = viveInitialRow:toVector3(15)
	local viveLeftArmPos = viveInitialRow:toVector3(9)
	local viveRightArmPos = viveInitialRow:toVector3(12)
	local viveUpVec = viveNeckPos - viveHipPos
	viveUpVec:normalize()
	local viveRightVec = viveRightArmPos - viveLeftArmPos
	viveRightVec:normalize()
	local loaderForwardVec = vector3()
	local viveForwardVec = vector3()
	loaderForwardVec:cross(loaderUpVec, loaderRightVec)
	loaderForwardVec = vector3(0,0,1)
	viveForwardVec:cross(viveUpVec, viveRightVec)
	-- compute arccos
	local angle =  math.acos(loaderForwardVec:dotProduct(viveForwardVec)/loaderForwardVec:length()*viveForwardVec:length())
	-- 각도만 구해서 방향을 모르는 상태, 두 벡터를 외적한 축을 기준으로 회전시키면 됨.
	local ax = vector3()
	ax:cross(viveForwardVec, loaderForwardVec)
	local tmpTr = transf(quater(angle, ax), vector3(0,0,0)) -- 초기 팔,다리 위치 틀어지면 조절, 파일마다 수동적으로 바꿔줘야함.
	local tr = vector3(1, 2.2, 3)
	for i=0, vive.pos:rows()-1 do
		for j=0, mocap.numTrackers-1 do 
			vive.pos:row(i):setVec3(j*3,  vive.pos:row(i):toVector3(j*3) + tr)
			--local tmpTr = transf(quater(angle, vector3(0,1,0)), vector3(0,0,0)) -- 초기 팔,다리 위치 틀어지면 조절, 파일마다 수동적으로 바꿔줘야함.
			-- facing direction align. mLoader와 바라보는 방향 같게 조정. 
			vive.pos:row(i):setVec3(j*3,  tmpTr*vive.pos:row(i):toVector3(j*3))
		end
	end
	]]
	--[[
	-- Change tracker order to match to model input sequence
	-- 왼손(0), 왼다리(3, 4), 머리(6, 8), 오른손(9, 12), 루트(12, 16), 오른다리(15, 20) 순서로 저장됨. (2023-04-11 촬영/viveraw_posori222.dat) 
	-- -> 루트(0), 왼다리(3, 4), 오른다리(6, 8), 왼손(9, 12), 오른손(12, 16), 머리(15, 20)
	local trackerHipPos = vive.pos:sub(0,0, 12,15)
	local trackerHipOri = vive.ori:sub(0,0, 16,20)
	local trackerLeftLegPos = vive.pos:sub(0,0, 3,6)
	local trackerLeftLegOri = vive.ori:sub(0,0, 4,8)
	local trackerRightLegPos = vive.pos:sub(0,0, 15,18)
	local trackerRightLegOri = vive.ori:sub(0,0, 20,24)
	local trackerLeftArmPos = vive.pos:sub(0,0, 0,3)
	local trackerLeftArmOri = vive.ori:sub(0,0, 0,4)
	local trackerRightArmPos = vive.pos:sub(0,0, 9,12)
	local trackerRightArmOri = vive.ori:sub(0,0, 12,16)
	local trackerHeadPos = vive.pos:sub(0,0, 6,9)
	local trackerHeadOri = vive.ori:sub(0,0, 8,12)

	-- 조립
	local datas = {}
	datas.pos = trackerHipPos..trackerLeftLegPos..trackerRightLegPos..trackerLeftArmPos..trackerRightArmPos..trackerHeadPos
    datas.ori = trackerHipOri..trackerLeftLegOri..trackerRightLegOri..trackerLeftArmOri..trackerRightArmOri..trackerHeadOri]]
	local datas = {}
	datas.pos = vive.pos
	datas.ori = vive.ori
	return datas
end





----------------------------------------------------------------------
