require('Model3')

Model2=LUAclass()
function Model2:__init() -- TF
	self.config=config
	self.past_window=PAST_WINDOW
	self.future_window=FUTURE_WINDOW
	self.delay=DELAY
end
function Model2:createPyModule()
	if TorchModule then
		modules['regress_pred']=TorchModule_siMLPe('TF/it40.ptl')
	end
	local dim=self:getInputFeatureDim()
	local odim=self:getOutputFeatureDim()
	python.F('TF.test','createModel',projectPath..'/TF/it40.pt', 'model2', 
	CT.vec(dim(1), odim(1), dim(0)))
end

function Model2:isUsingFilteredReferenceFrame()
	return false
end

function Model2:getInputFeatureDim()
	local numTracker=#self.config.trackerBones
	return CT.ivec(self.past_window-1, numTracker*9)
end
function Model2:getOutputFeatureDim()
	if not mLoader then
		print("Error! mLoader is nil")
		return nil
	end
	local numBone=mLoader:numBone()
	return CT.ivec(self.future_window-1, (numBone-1)*6+3)
end

function Model2:predictPose(refCoord, _inputFeature)
	local inputFeature=_inputFeature:sub(-self.past_window+1, 0)
	local dim=self:getOutputFeatureDim()
	local outputFeature=matrixn(dim(0), dim(1))
	-- prediction only
	python.F('TF.test','test_TF',inputFeature, outputFeature, 'model2')

	--local poseFeature=vectorn() getPoseFeature(refCoord, poseFeature)
	local poseFeature=outputFeature:row(irow or 0)

	local pose=featureToPose(refCoord, poseFeature)
	local posedof=vectorn()
	mLoader.dofInfo:getDOF(pose, posedof)
	return posedof
end
function Model2:getPose(irow)
	local trackerHistory=tracker_history

	assert(irow==0) -- other cases : not implemented yet
	local inputFeature=_getInputFeatureNoFilter(trackerHistory, trackerHistory:rows())

	local refCoord=trackerHistory:row(trackerHistory:rows()-self.delay):toTransf():project2D()
	return self:predictPose(refCoord, inputFeature:sub(-self.past_window+1, 0))
end
