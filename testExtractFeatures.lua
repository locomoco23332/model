
function ctor()
	setView()

	mEventReceiver = EVR()
	if RE.hasGUI() then 
		local osm =RE.ogreSceneManager()
		osm:setFogNone()
		draw = true
	end

	--mLoader=MainLib.VRMLloader ("../Resource/motion/locomotion_hyunwoo/hyunwoo_lowdof_T_boxfoot.wrl")
	mFbxLoader=FBXloader (config.loaderFile) mLoader=mFbxLoader.loader
	mMotion=Motion(mLoader)
	mMotion:importBinary(config.motFile)

	for i, bone in ipairs(config.trackerBones) do
		local bone2=mLoader:getBoneByName(bone.name)
		bone.idx=bone2:treeIndex()
	end

	if mFbxLoader then
		mSkin1 = RE.createFBXskin(mFbxLoader , false)
	else
		mSkin1 = RE.createVRMLskin(mLoader , false)
	end
	mSkin1:setScale(config.skinScale,config.skinScale,config.skinScale)
	mSkin1:setMaterial('lightgrey_transparent')

	this:updateLayout()
	iframe = 0
	
	tracker_data = All_tracker_data()


	--inputFeatures=hypermatrixn()
	--outputFeatures=hypermatrixn()


	print('refCoord')
	refCoord=matrixn(mMotion:numFrames(), 7)
	for iframe=0, mMotion:numFrames()-1 do
		if iframe>PAST_WINDOW then
			if (iframe+1)%5000==0 then
				print(iframe,'/',mMotion:numFrames())
			end
			refCoord:row(iframe):setTransf(0, getRefCoord(tracker_data, iframe))
		end
	end
	--assert(inputFeatures:pages()==outputFeatures:pages())
	local numBone=mLoader:numBone()
	poseFeatureCache=matrixn(mMotion:numFrames(), (numBone-1)*6+3)

	local referenceFrame=transf()
	referenceFrame:identity()
	for i=0, mMotion:numFrames()-1 do
		mLoader:setPose(mMotion:pose(i))
		getPoseFeature(referenceFrame,poseFeatureCache:row(i))
		if i%5000==0 then
			print(i)
		end
	end

	local file=util.BinaryFile()
	file:openWrite('features.dat');
	--file:pack(inputFeatures)
	--file:pack(outputFeatures)
	file:pack(tracker_data)
	file:pack(refCoord)

	local validFrames=listValidFrames(PAST_WINDOW, DELAY, FUTURE_WINDOW)

	file:pack(validFrames)

	file:pack(poseFeatureCache)
	file:close()

	print("features.dat exported to "..os.currentDirectory().."!!!")
	print("you can close this window now")
	--this('exit!',0)
end
elapsedTime = 0
function frameMove(fElapsedTime)

	elapsedTime=elapsedTime+fElapsedTime
	if elapsedTime>1/30 then
		iframe=iframe+1
		getTrackerData(iframe,draw)

		if iframe>PAST_WINDOW and iframe<4000 then
			getRefCoord(tracker_data, iframe, true)


		end
		local pose = mMotion:pose(iframe)
		mSkin1:setPose(pose)
		mLoader:setPose(pose)

		elapsedTime=math.min(elapsedTime-1/30, 2/30)
	end
	return 0
end
function onCallback(w, userData)
end
function handleRendererEvent(ev, button, x, y)
	return 0
end

function onCallback(w, userData)
end
