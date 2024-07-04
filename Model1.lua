require('Model3')

Model1=LUAclass()
function Model1:__init() -- AE10
	--local projectPath=os.parentPath(os.currentDirectory())
	--package.path=package.path..';'..projectPath..'/lua/?.lua'
	--ctorFromFileIn(projectPath)
	--ctor_part1(projectPath) 
	--
	self.config=config
	self.past_window=11 -- ==> so that condition_frame==10
	self.future_window=11 --> so that output_numframe==10
	self.delay=10
	-- global history buffer should be larger
	assert(self.past_window<=PAST_WINDOW)
	assert(self.future_window<=FUTURE_WINDOW)
	assert(self.delay<=DELAY)
end
function Model1:createPyModule()
	if TorchModule then
		modules['regress_pred']=TorchModule_siMLPe('AE/TrackerAE10.ptl')
	end
	python.F('AE.test','createModel',projectPath..'/AE/TrackerAE10.pt', 'model1')
end

function Model1:isUsingFilteredReferenceFrame()
	return false
end

function Model1:getInputFeatureDim()
	local numTracker=#self.config.trackerBones
	return CT.ivec(self.past_window-1, numTracker*9)
end
function Model1:getOutputFeatureDim()
	if not mLoader then
		print("Error! mLoader is nil")
		return nil
	end
	local numBone=mLoader:numBone()
	return CT.ivec(self.future_window-1, (numBone-1)*6+3)
end

function Model1:predictPose(refCoord, _inputFeature)
	local inputFeature=_inputFeature:sub(-self.past_window+1, 0)
	local dim=self:getOutputFeatureDim()
	local outputFeature=matrixn(dim(0), dim(1))
	-- prediction only
	python.F('AE.test','test_model_mat',inputFeature, outputFeature, 'model1')

	--local poseFeature=vectorn() getPoseFeature(refCoord, poseFeature)
	local poseFeature=outputFeature:row(irow or 0)

	local pose=featureToPose(refCoord, poseFeature)
	local posedof=vectorn()
	mLoader.dofInfo:getDOF(pose, posedof)
	return posedof
end
function Model1:getPose(irow)
	local trackerHistory=tracker_history

	assert(irow==0) -- other cases : not implemented yet
	local inputFeature=_getInputFeatureNoFilter(trackerHistory, trackerHistory:rows())

	local refCoord=trackerHistory:row(trackerHistory:rows()-self.delay):toTransf():project2D()
	return self:predictPose(refCoord, inputFeature:sub(-self.past_window+1, 0))
end
