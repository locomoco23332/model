-- use tp script
require("config")
package.projectPath='../Samples/classification/'
package.path=package.path..";../Samples/classification/lua/?.lua" --;"..package.path
require("common")
require("module")
-- require('subRoutines/classifyLib')
fc=require('retargetting/module/footSkateCleanup')
require('subRoutines/Timeline')
FBXloader=require('FBXloader')
require("RigidBodyWin/subRoutines/Constraints")
require("RigidBodyWin/subRoutines/CollisionChecker")
config = {skinScale = 100}
fc=require('retargetting/module/footSkateCleanup')
config=nil

scriptPath=util.getScriptPath()
projectPath=os.parentPath(scriptPath)

package.path=package.path..';'..scriptPath..'/?.lua'

require("configVIVE")

function  changeTrackerRotation(tracker_data_i, refCoord)
	local numTracker=#config.trackerBones
	local changedTrackerData = vectorn(numTracker * 9)
	for j=0,numTracker-1 do 
		local tf=tracker_data_i:toTransf(7*j)
		tf=refCoord:toLocal(tf)
		changedTrackerData:setTransf9(9*j, tf)
	end
	return changedTrackerData
end
function getInputFeature(iframe)
	local referenceFrame=refCoord:row(iframe):toTransf()
	return _getInputFeature(tracker_data, iframe, referenceFrame), referenceFrame
end
function getInputFeatureNoFilter(iframe)
	return _getInputFeatureNoFilter(tracker_data, iframe)
end
function _getInputFeatureNoFilter(tracker_data, iframe)
	local numTracker=#config.trackerBones
	local s=iframe-PAST_WINDOW+1
	local e=iframe
	local ifeature=matrixn(PAST_WINDOW-1, numTracker*9)
	for i=s,e-1 do
		local referenceFrame=tracker_data:row(i-1):toTransf(0):project2D()
		-- 고정된 reference frame사용!
		ifeature:row(i-s):assign(changeTrackerRotation(tracker_data:row(i), referenceFrame))
	end
	assert(not ifeature:isnan())
	return ifeature

end
function _getInputFeature(tracker_data, iframe, referenceFrame)
	local numTracker=#config.trackerBones
	local s=iframe-PAST_WINDOW
	local e=iframe
	local ifeature=matrixn(PAST_WINDOW, numTracker*9)
	for i=s,e-1 do
		-- 고정된 reference frame사용!
		ifeature:row(i-s):assign(changeTrackerRotation(tracker_data:row(i), referenceFrame))
	end
	assert(not ifeature:isnan())
	return ifeature
end
function getOutputFeature(iframe)
	local s=iframe-DELAY
	local e=iframe-DELAY+FUTURE_WINDOW
	local numBone=mLoader:numBone()
	local ofeature=matrixn(FUTURE_WINDOW, (numBone-1)*6+3)

	local referenceFrame=refCoord:row(iframe):toTransf()
	for i=s,e-1 do
		ofeature:row(i-s):assign(poseFeatureCache:row(i))
		local root=mMotion:pose(i):rootTransformation()
		root=referenceFrame:toLocal(root)
		ofeature:row(i-s):setTransf9(0,root)
	end
	assert(not ofeature:isnan())
	return ofeature
end
function getOutputFeatureNoFilter(iframe)
	local s=iframe-DELAY+1
	local e=iframe-DELAY+FUTURE_WINDOW
	local numBone=mLoader:numBone()
	local ofeature=matrixn(FUTURE_WINDOW-1, (numBone-1)*6+3)

	for i=s,e-1 do
		local referenceFrame=tracker_data.row(i-1):toTransf():project2D()
		ofeature:row(i-s):assign(poseFeatureCache:row(i))
		local root=mMotion:pose(i):rootTransformation()
		root=referenceFrame:toLocal(root)
		ofeature:row(i-s):setTransf9(0,root)
	end
	assert(not ofeature:isnan())
	return ofeature
end



function setView()

	RE.viewpoint():setFOVy(45.000000)
	RE.viewpoint():setZoom(1.000000)
	RE.viewpoint().vpos:set(319.085224, 331.752372, 773.993372)
	RE.viewpoint().vat:set(200.404942, -142.968756, 61.911680)
	RE.viewpoint():update()
end
function loadFeaturesToLuaGlobals()
	mFbxLoader=FBXloader (config.loaderFile) mLoader=mFbxLoader.loader

	mMotion=Motion(mLoader)
	mMotion:importBinary(config.motFile)

	rootTraj=matrixn(mMotion:numFrames(), 7)
	for i=0, mMotion:numFrames()-1 do
		rootTraj:row(i):setTransf(0, mMotion:pose(i):rootTransformation())
	end

	local file=util.BinaryFile()
	tracker_data = matrixn()
	refCoord=matrixn()
	validFrames=intvectorn()
	poseFeatureCache=matrixn()
	file:openRead('features.dat');
	--file:pack(inputFeatures)
	--file:pack(outputFeatures)
	file:unpack(tracker_data)
	file:unpack(refCoord)
	file:unpack(validFrames)
	file:unpack(poseFeatureCache)
	file:close()
end

function getPoseFeature(referenceFrame, output)
	local numBone=mLoader:numBone()
	local fk=mLoader:fkSolver()
	output:setSize((numBone-1)*6+3)

	local root=fk:globalFrame(1)
	root=referenceFrame:toLocal(root)

	output:setTransf9(0, root)
	for i=2, mLoader:numBone()-1 do
		output:setQuater6((i-2)*6+9, fk:localFrame(i).rotation)
	end

	return output
end

function featureToPose(refCoord, poseFeature)
	local fk=mLoader:fkSolver()
	local root=refCoord*poseFeature:toTransf9(0)
	fk:localFrame(1):assign(root)
	for i=2, mLoader:numBone()-1 do
		fk:localFrame(i).rotation:assign( poseFeature:toQuater6((i-2)*6+9))
	end
	local pose=Pose()
	fk:getPoseFromLocal(pose)
	return pose
end

function listValidFrames(PAST_WINDOW, DELAY, FUTURE_WINDOW, step)
	if not step then step=1 end
	local featureInvalid=mMotion:getDiscontinuity()

	if false then
		-- 위랑 같은 뜻.
		local segFinder=SegmentFinder(mMotion:getDiscontinuity())

		for iseg=0, segFinder:numSegment()-1 do
			-- 여러 모션의 시작 프레임과 끝 프레임
			local startframe=segFinder:startFrame(iseg)
			local endframe=segFinder:endFrame(iseg)

			featureInvalid:range(startframe+1, endframe):setAllValue(false)
		end
	end

	local validFrames=intvectorn()
	--validFrames:reserve(featureInvalid:size())
	for iframe=0, featureInvalid:size()-1, step do

		local s=iframe-PAST_WINDOW
		local e=iframe-DELAY+FUTURE_WINDOW

		if s>0 and e<=featureInvalid:size() then
			if featureInvalid:range(s,e):count()==0 then
				validFrames:pushBack(iframe)
				--inputFeatures:pushBack(getInputFeature(iframe))
				--outputFeatures:pushBack(getOutputFeature(iframe))
			end
		end
	end
	return validFrames
end


function dtor()
end


if EventReceiver then 
	EVR = LUAclass(EventReceiver)
	function EVR:__init(graph)
		self.currFrame = 0
		self.cameraInfo = {}
	end 
end	

function EVR:onFrameChanged(win, iframe)
end

function All_tracker_data()
	local data = matrixn(mMotion:numFrames(),42)
	for i=0,data:rows()-1 do
		local trackerData=getTrackerData(i)
		if (i+1)%5000==0 then
			print(i,'/',mMotion:numFrames())
		end
		for j=0,5 do

			if useNoise and i%500 == 0 then 
				local r = random.randint(0,5)
				if r == j then 
					data:row(i):setVec3(7*j,vector3(0,0,0))
					data:row(i):setQuater(7*j+3,quater(1,0,0,0))
				else 
					data:row(i):setVec3(7*j,trackerData.pos(j))
					data:row(i):setQuater(3+7*j,trackerData.ori(j))
				end
			else
				data:row(i):setVec3(7*j,trackerData.pos(j))
				data:row(i):setQuater(3+7*j,trackerData.ori(j))
			end
		end
	end
	---------- change local coord ------------ 
	--[[
	local local_data = matrixn(data:rows(),data:cols()+6*6)
	for i=0,data:rows()-2 do 
		local_data:row(i):assign(changeTrackerLocal(data:row(i):range(0,7),data:row(i),data:row(i+1)))
	end
	local final_data = matrixn(data:rows(),local_data:cols()+12)
	final_data:assign(changeTrackerRotation(local_data))
	]]
	return data 
end


function All_data_frames()
	local data = matrixn(mMotion:numFrames()-1,mMotion:numDOF()-7)
	for i=0,data:rows()-1 do 
		data:row(i):assign(mMotion:row(i):range(7,mMotion:numDOF()))
	end
	return data
end



--discontinuity check하기--
function getDiscontinuity()
	local discont = vectorn()
	local conFile=util.BinaryFile()
	conFile:openRead('../Resource/motion/locomotion_hyunwoo/hyunwoo_lowdof_T_locomotion_hl.dof.discont')
	discont=conFile:unpackAny() -- discontinuity
  	local  discont_frames = vectorn()
	-- for i=0,discont:size()-1 do
	-- 	if discont(i) == true then 
	-- 		discont_frames:pushBack(i)
	-- 	end
	-- end
  	--manually checked discotinuity stitch motion --
	-- discont_frames:pushBack(28081)
	-- discont_frames:pushBack(29572)
	-- discont_frames:pushBack(29788)
	-- discont_frames:pushBack(33835)
	-- discont_frames:pushBack(42557)
	discont_frames:pushBack(0)
	discont_frames:pushBack(mMotion:numFrames())
	return discont_frames
end




function pushBack(matrix,data)
	for i=0,matrix:rows()-2 do
		matrix:row(i):assign(matrix:row(i+1))
	end
	matrix:row(matrix:rows()-1):assign(data)
end


function getTrackerData(iframe,draw)
	--root,leftAnkle,rightAnkle,leftElbow,righrElbow,neck
	mLoader:setPose(mMotion:pose(iframe))
	local n=#config.trackerBones
	local pos = vector3N(n)
	local ori = quaterN(n)

	for ii, bone in ipairs(config.trackerBones) do
		local j= ii-1
		local transform = mLoader:bone(bone.idx):getFrame() *bone.trackerTransf
		pos(j):assign(transform.translation)
		ori(j):assign(transform.rotation)
		if draw then 
			dbg.draw('Axes',transform,''..ii,100)
		end
	end
	return {pos=pos, ori=ori}
end









function getCenterCoord(a,b)
	assert(lunaType(a)=='transf','a is not trnasf')
	assert(lunaType(b)=='transf','b is not transf')
	local output = transf()
	output:interpolate(0.5,a,b)
	return output 
end


function getRefCoord(tracker_data, iframe, draw)
	local kernel=vectorn()
	math.getGaussFilter(PAST_WINDOW, kernel)
	local center=math.filterSingle(tracker_data:sub(iframe-PAST_WINDOW,iframe,0,3), PAST_WINDOW)
	local centerQ=quater()
	centerQ:blend(kernel, tracker_data:sub(iframe-PAST_WINDOW,iframe,3,7))

	local tf=transf()
	tf.translation:assign(center:toVector3(0))
	tf.translation.y=0
	tf.rotation:assign(centerQ:Normalize():rotationY())
	if draw then
		dbg.draw('Axes',tf,'refcoord',100)
	end
	return tf
end



