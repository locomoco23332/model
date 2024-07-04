
function createIKsolver()
	mEffectors=MotionUtil.Effectors()
	mEffectors:resize(4)
	local kneeIndex=intvectorn(4)
	local axis=CT.vec(1,1,-1,1)
	for i=2,5 do
		local binfo=config.trackerBones[i]

		kneeIndex:set(i-2, binfo.idx)
		local eebone=mLoader:bone(binfo.idx+1)
		local localpos=vector3(0,0,0)
		mEffectors(i-2):init(eebone,localpos)
	end

	mLoader:updateInitialBone()
	mInitialPose=mLoader:getPoseDOF()
	mSolver=LimbIKsolver(mLoader.dofInfo,mEffectors, kneeIndex, axis)
end
function solveIK(poseInout, td, solveSwivelAngle)
	local conpos=vector3N(4)
	local conori=quaterN(4)
	-- root*trackerOffset=trackerData
	local newRootTF=td:toTransf(0)*config.trackerBones[1].trackerTransf:inverse()
	for i=0,3 do
		-- parentGlobal*selfOffset=selfG
		-- parentG*trackerOffset=trackerData
		-- selfG*selfOffset:inverse()*trackerOffset=trackerData
		--> selfG=trackerData*trackerOffset:inverse()*selfOffset
		local eeTF=td:toTransf(7*(i+1))*config.trackerBones[i+2].trackerTransf:inverse()*mEffectors(i).bone:getOffsetTransform()

		conpos(i):assign(eeTF.translation)
		conori(i):assign(eeTF.rotation)
	end


	mSolver:IKsolve3(poseInout, newRootTF, conpos, conori, CT.vec(1,1,1,1))

	if solveSwivelAngle then
		-- further solve elbow-swivel angle
		mLoader:setPoseDOF(poseInout)
		for i=0, 3 do
			local boneInfo=config.trackerBones[i+2]
			local elb_bone=mLoader:bone(boneInfo.idx)
			local sh_bone=elb_bone:parent()
			local wrist_bone=elb_bone:childHead()
			local trackerData=td:toTransf(7*(i+1))
			local trackerOffset=boneInfo.trackerTransf
			-- parentG*trackerOffset=trackerData
			local goal_elb=(trackerData*trackerOffset:inverse()).translation
			IKsolveSwivelAngle(mLoader, sh_bone, elb_bone, wrist_bone, goal_elb)
		end
		for i=0, 3 do
			local boneInfo=config.trackerBones[i+2]
			local elb_bone=mLoader:bone(boneInfo.idx)
			local wrist_bone=elb_bone:childHead()
			wrist_bone:getLocalFrame().rotation:identity()
		end
		mLoader:fkSolver():getPoseDOFfromLocal(poseInout)
	end

	return poseInout
end
function IKsolveSwivelAngle(loader, sh_bone, elb_bone, wrist_bone, goal_elb)
	local p0=sh_bone:getFrame().translation:copy()
	local p1=elb_bone:getFrame().translation:copy()
	local p2=wrist_bone:getFrame().translation:copy()
	
	local axis=p2-p0
	axis:normalize()
	local center=(p0+p2)/2
	local front=p1-center
	local target=goal_elb-center
	local q=quater()
	q:setAxisRotation(axis, front, target)
	local angle=q:rotationAngleAboutAxis(axis)
	if angle<-math.rad(30) then
		angle=-math.rad(30)
	elseif angle>math.rad(30) then
		angle=math.rad(30)
	end
	local femurLen=(p1-p0):length()
	local importance=sop.clampMap(target:length(), 0.2*femurLen,0.4*femurLen, 0, 1) 	
	angle=angle*importance
	q:setRotation(axis, angle)
	--q:setRotation(axis, math.rad(90))
	loader:rotateBoneGlobal(sh_bone,q)
end
