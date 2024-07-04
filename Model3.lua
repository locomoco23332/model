

Model3=LUAclass()
function Model3:__init(config)
	--local projectPath=os.parentPath(os.currentDirectory())
	--package.path=package.path..';'..projectPath..'/lua/?.lua'
	--ctorFromFileIn(projectPath)
	--ctor_part1(projectPath) 
	--
	self.config=config
end
function Model3:createPyModule()
	if TorchModule then
		modules['regress_pred']=TorchModule_siMLPe('siMLPe/model-iter-10000-derivInput.ptl')
	end
	python.F('siMLPe.test','createModel',projectPath..'/siMLPe/model-iter-40000.pth')
end

function Model3:getInputFeatureDim()
	local numTracker=#self.config.trackerBones
	return CT.ivec(PAST_WINDOW, numTracker*9)
end
function Model3:getOutputFeatureDim()
	if not mLoader then
		print("Error! mLoader is nil")
		return nil
	end
	local numBone=mLoader:numBone()
	return CT.ivec(FUTURE_WINDOW, (numBone-1)*6+3)
end

function Model3:isUsingFilteredReferenceFrame()
	return true
end

function Model3:predictPose(refCoord, inputFeature)
	local outputFeature=matrixn()
	if true then
		-- prediction only
		python.F('siMLPe.test','regress_pred',inputFeature, outputFeature)
	else
		-- evaluation also
		local inputFeature2=getInputFeature(iframe)
		local outputFeature_gt=getOutputFeature(iframe)
		python.F('siMLPe.test','regress_pred_eval',inputFeature, outputFeature, outputFeature_gt)
	end
	if outputFeature:rows()==0 then
		error('Error! nopython and no torch.')
	end

	--local poseFeature=vectorn() getPoseFeature(refCoord, poseFeature)
	local poseFeature=outputFeature:row(irow or 0)

	local pose=featureToPose(refCoord, poseFeature)
	local posedof=vectorn()
	mLoader.dofInfo:getDOF(pose, posedof)
	return posedof
end
function Model3:getPose(irow)
	local trackerHistory=tracker_history

	local refCoord=getRefCoord(trackerHistory, trackerHistory:rows())
	local inputFeature=_getInputFeature(trackerHistory, trackerHistory:rows(), refCoord)

	return self:predictPose(refCoord, inputFeature)
end

Queue=LUAclass()
function Queue:__init(n)
	self.n=n
	self.data={}
	self.front=1
end
function Queue:pushBack(data)
	if #self.data==self.n then
		self.data[self.front]=data
		self.front=self.front+1
		if self.front>self.n then
			self.front=1
		end
	else
		table.insert(self.data, data)
	end
end
function Queue:back()
	local f=self.front
	if f==1 then
		return self.data[#self.data]
	else
		return self.data[f-1]
	end
end
-- i in [0, n-1]
function Queue:getElt(i)
	return self.data[math.fmod(i+self.front-1, self.n)+1]
end

function Queue:front()
	return self.data[self.front]
end

OnlineFilter=LUAclass()

function OnlineFilter:__init(loader, pose, filterSize, useMitchellFilter)
	self.filterSize=filterSize
	self.loader=loader
	self.queue=Queue(filterSize)
	if pose then
		self:setCurrPose(pose)
	end
	self.useMitchellFilter=useMitchellFilter or false
end

function OnlineFilter:setCurrPose(pose)
	self.queue:pushBack(pose:copy())
end

function OnlineFilter:getFiltered()
	if #self.queue.data==self.queue.n then

		local sum=vectorn(self.queue:back():size())
		sum:setAllValue(0)
		if true then
			-- use gaussian filter for joint angles
			local arrayV=matrixn(self.queue.n, self.queue:back():size()-7)
			for i=0, arrayV:size()-1 do
				arrayV:row(i):assign(self.queue:getElt(i):slice(7,0))
			end
			local v=math.filterSingle(arrayV, self.queue.n, self.useMitchellFilter)
			sum:slice(7,0):assign(v)
		else

			for i,v in ipairs(self.queue.data) do
				sum:radd(self.queue.data[i])
			end
			sum:rmult(1/self.queue.n)
		end
		if false then
			-- simple renormalization works only when filter size is small
			sum:setQuater(3, sum:toQuater(3):Normalize())
		else
			-- use gaussian filter for root pos and ori.
			local arrayQ=quaterN(self.queue.n)
			local arrayV=vector3N(self.queue.n)
			for i=0, arrayQ:size()-1 do
				arrayQ(i):assign(self.queue:getElt(i):toQuater(3))
				arrayV(i):assign(self.queue:getElt(i):toVector3(0))
			end
			--math.filterQuat(arrayQ:matView(), self.queue.n)
			--sum:setQuater(3, arrayQ(math.floor(self.queue.n/2)))
			local q=math.filterQuatSingle(arrayQ:matView(), self.queue.n, self.useMitchellFilter)
			local v=math.filterSingle(arrayV:matView(), self.queue.n, self.useMitchellFilter)
			sum:setVec3(0, v:toVector3())
			sum:setQuater(3, q:toQuater())
		end
		return sum
	else
		return self.queue:back()
	end
end

