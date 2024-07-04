function evaluation(GTMotionDOF, outMotionDOF, loader)
    assert(GTMotionDOF:rows() == outMotionDOF:rows())
    local fkSolver = loader:fkSolver()
	
    local GTGlobalPos = matrixn(GTMotionDOF:rows(), 3*(loader:numBone()-1))
    local outGlobalPos = matrixn(GTMotionDOF:rows(), 3*(loader:numBone()-1))
    local GTLocalOri = matrixn(GTMotionDOF:rows(), 4*(loader:numBone()-1))
    local outLocalOri = matrixn(GTMotionDOF:rows(), 4*(loader:numBone()-1))
    
    GTGlobalPos:setAllValue(0)
    outGlobalPos:setAllValue(0)
    GTLocalOri:setAllValue(0)
    outLocalOri:setAllValue(0)

    for i=0, GTMotionDOF:rows()-1 do
        fkSolver:setPoseDOF(GTMotionDOF:row(i))
        for j=1, loader:numBone()-1 do
            GTGlobalPos:row(i):setVec3(3*(j-1), fkSolver:globalFrame(j).translation)
            GTLocalOri:row(i):setQuater(4*(j-1), fkSolver:localFrame(j).rotation)
        end
        fkSolver:setPoseDOF(outMotionDOF:row(i))
        for j=1, loader:numBone()-1 do
            outGlobalPos:row(i):setVec3(3*(j-1), fkSolver:globalFrame(j).translation)
            outLocalOri:row(i):setQuater(4*(j-1), fkSolver:localFrame(j).rotation)
        end
    end

    local mpjpe = evalMPJPE(GTGlobalPos, outGlobalPos, loader)
    local mpjre = evalMPJRE(GTLocalOri, outLocalOri, loader)
    local GTJitter = evalJitter(GTGlobalPos, loader)
    local outJitter = evalJitter(outGlobalPos, loader)
    --
    --dbg.console()
    local evals = {mpjpe=mpjpe, mpjre=mpjre, GTJitter=GTJitter, outJitter=outJitter}
	return evals
end

function evalMPJPE(GTPosMatrixN, outPosMatrixN, loader)
    -- in cm unit
    local numBone = loader:numBone()-1
    local mpjpe = 0.0
    assert(GTPosMatrixN:rows() == outPosMatrixN:rows())

    for i=0, GTPosMatrixN:rows()-1 do
        local GTPositions = GTPosMatrixN:row(i)
        local outPositions = outPosMatrixN:row(i)
        local GTRootPos = GTPositions:toVector3(0)
        local outRootPos = outPositions:toVector3(0)
        for j=2, numBone do
            local tmpGTBonePos = GTPositions:toVector3(3*(j-1))
            local tmpOutBonePos = outPositions:toVector3(3*(j-1))
            local GTRelRootPos = tmpGTBonePos - GTRootPos -- relative to root
            local outRelRootPos = tmpOutBonePos - outRootPos
			local a =GTRelRootPos:distance(outRelRootPos) 
			--print(a)
			if a  > 1000 then 
				dbg.console()
			end
			--print(GTRelRootPos:distance(outRelRootPos))
            mpjpe = mpjpe + GTRelRootPos:distance(outRelRootPos)
        end
    end
    
    mpjpe = mpjpe / (numBone-1)  -- except root
    mpjpe = mpjpe / GTPosMatrixN:rows()
    mpjpe = mpjpe * 100 -- m to cm
    return mpjpe
end

function evalMPJRE(GTOriMatrixN, outOriMatrixN, loader)
    -- in degree unit
    local numBone = loader:numBone()-1
    local mpjre = 0.0
    assert(GTOriMatrixN:rows() == outOriMatrixN:rows())

    for i=0, GTOriMatrixN:rows()-1 do
        local GTOris = GTOriMatrixN:row(i)
        local outOris = outOriMatrixN:row(i)
        for j=1, numBone do
            local tmpGTBoneOri = GTOris:toQuater(4*(j-1))
            local tmpOutBoneOri = outOris:toQuater(4*(j-1))
            local diff = quater()
            diff:difference(tmpGTBoneOri, tmpOutBoneOri)
			diff:normalize()
            mpjre = mpjre + diff:rotationAngle()
        end
    end
    mpjre = mpjre / numBone
    mpjre = mpjre / GTOriMatrixN:rows()
    mpjre = mpjre * 180 / math.pi -- rad to deg
    return mpjre
end

function evalJitter(posMatrixN, loader)
    -- in 10^2m/s^3
    local numBone = loader:numBone()-1
    local jitter = 0.0

    local slice1 = posMatrixN:slice(3,0, 0,0)
    local slice2 = posMatrixN:slice(2,-1, 0,0)
    local slice3 = posMatrixN:slice(1,-2, 0,0)
    local slice4 = posMatrixN:slice(0,-3, 0,0)
    local jerk = slice1 -3 * slice2 + 3 * slice3 - slice4
    local size = jerk:rows()

    for i=0, size-1 do
        local tmpJerk = jerk:row(i)
        for j=1, numBone do
            local tmpBoneJerk = tmpJerk:toVector3(3*(j-1))
            jitter = jitter + tmpBoneJerk:length()
        end
    end
    
    jitter = jitter / numBone
    jitter = jitter / size
    jitter = jitter * 100.0 -- m/s^3 -> 10^2m/s^3
    return jitter 
end
