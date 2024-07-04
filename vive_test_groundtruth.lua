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
	loadFeaturesToLuaGlobals()

	mLoader2=mLoader -- finger 아직 안씀.

	for i, bone in ipairs(config.trackerBones) do
		local bone2=mLoader2:getBoneByName(bone.name)
		bone.idx=bone2:treeIndex()
		bone.startT=mLoader2.dofInfo:startT(bone2:treeIndex()) -- posedof의 스타트 인덱스. 
		bone.idx1=mLoader:getBoneByName(bone.name):treeIndex()
	end

	mSkinGT = RE.createFBXskin(mFbxLoader,false)
	mSkinGT:setScale(config.skinScale,config.skinScale,config.skinScale)
	mSkinGT:setTranslation(100,0,0)
	mSkinGT:setMaterial('lightgrey_transparent')

	--DOFcontainer = MotionDOFcontainer(mLoader2.dofInfo , "../seok_motion/stitchMotion_mixamo.dof")
	--mMotionDOF = DOFcontainer.mot
	mSkin2 = RE.createFBXskin(mFbxLoader,false)
	mSkin2:setScale(config.skinScale,config.skinScale,config.skinScale)

	this:addButton('init AE10 (model1)')
	this:addButton('init TF (model2)')
	this:addButton('init siMLPe (model3)')
	this:addButton('init siMLPe (model4)')
	this:updateLayout()

	isValidFrame=boolN(mMotion:numFrames())
	isValidFrame:setAllValue(false)
	for i=0, validFrames:size()-1 do
		isValidFrame:set(validFrames(i), true)
	end

	mTimeline=Timeline('timeline', mMotion:numFrames(), config.rendering_step)

	get_delta = true -- calibrate
	use_siMLPe4()
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
	end

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
	if iframe>=mMotion:numFrames() then return end

	mSkinGT:setPose(mMotion:pose(iframe))

	if not isValidFrame(iframe) then return end

	local function  drawTrackers(tracker_data_i)
		local numTracker=#config.trackerBones
		for j=0,numTracker-1 do 
			local tf=tracker_data_i:toTransf(7*j)
			dbg.draw('Axes',tf,'tracker'..j,100)
		end
	end
	drawTrackers(tracker_data:row(iframe-DELAY-1))

	local pose
	if model:isUsingFilteredReferenceFrame() then
		local inputFeature, referenceFrame=getInputFeature(iframe)
		pose=model:predictPose(referenceFrame, inputFeature)
	else
		local inputFeature=getInputFeatureNoFilter(iframe)
		local prev_referenceFrame=tracker_data:row(iframe-DELAY):toTransf(0):project2D()
		pose=model:predictPose(prev_referenceFrame, inputFeature) -- pose corresponds to frame =(iframe-DELAY+1)
	end

	mSkin2:setPoseDOF(pose)
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

elapsedTime = 0
function frameMove(fElapsedTime)
end

