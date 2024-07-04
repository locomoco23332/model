require('Model3')

Model4=LUAclass()
function Model4:__init()
	--local projectPath=os.parentPath(os.currentDirectory())
	--package.path=package.path..';'..projectPath..'/lua/?.lua'
	--ctorFromFileIn(projectPath)
	--ctor_part1(projectPath) 
	--
	self.config=config
end
function Model4:createPyModule()
	if TorchModule then
		modules['regress_pred']=TorchModule_siMLPe('siMLPe/model-iter-10000-derivInput.ptl')
	end
	python.F('siMLPe.test','createModel',projectPath..'/siMLPe/model4-iter-5000.pth', 'model4')
end

function Model4:isUsingFilteredReferenceFrame()
	return false
end

function Model4:getInputFeatureDim()
	local numTracker=#self.config.trackerBones
	return CT.ivec(PAST_WINDOW-1, numTracker*9)
end
function Model4:getOutputFeatureDim()
	if not mLoader then
		print("Error! mLoader is nil")
		return nil
	end
	local numBone=mLoader:numBone()
	return CT.ivec(FUTURE_WINDOW-1, (numBone-1)*6+3)
end

function Model4:predictPose(refCoord, inputFeature)
	local outputFeature=matrixn()
	-- prediction only
	python.F('siMLPe.test','regress_pred',inputFeature, outputFeature, 'model4')

	--local poseFeature=vectorn() getPoseFeature(refCoord, poseFeature)
	local poseFeature=outputFeature:row(irow or 0)

	local pose=featureToPose(refCoord, poseFeature)
	local posedof=vectorn()
	mLoader.dofInfo:getDOF(pose, posedof)
	return posedof
end
function Model4:getPose(irow)
	local trackerHistory=tracker_history

	assert(irow==0) -- other cases : not implemented yet
	local inputFeature=_getInputFeatureNoFilter(trackerHistory, trackerHistory:rows())

	local refCoord=trackerHistory:row(trackerHistory:rows()-DELAY):toTransf():project2D()
	return self:predictPose(refCoord, inputFeature)
end
