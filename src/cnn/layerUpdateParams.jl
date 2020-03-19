
#TODO make sure to renew V and S at each epoch 

function layerUpdateParams!(
                       model::Model,
                       cLayer::CL,
                       cnt::Integer = -1;
                       tMiniBatch::Integer = 1,
                       ) where {CL <: ConvLayer}

   optimizer = model.optimizer
   α = model.α
   β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

   if cLayer.updateCount >= cnt
       return nothing
   end #if cLayer.updateCount >= cnt

   if !all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
       return nothing
   end

   #initialize the needed variables to hold the corrected values
   #it is being init here cause these are not needed elsewhere

   if optimizer==:adam || optimizer==:momentum
       VCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))

       cLayer.V[:dw] .= β1 .* cLayer.V[:dw] .+ (1-β1) .* cLayer.dW
       cLayer.V[:db] .= β1 .* cLayer.V[:db] .+ (1-β1) .* cLayer.dB

       ##correcting
       VCorrected[:dw] .= cLayer.V[:dw] ./ (1-β1^tMiniBatch)
       VCorrected[:db] .= cLayer.V[:db] ./ (1-β1^tMiniBatch)

       if optimizer==:adam
           SCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))
           cLayer.S[:dw] .= β2 .* cLayer.S[:dw] .+ (1-β2) .* (cLayer.dW.^2)
           cLayer.S[:db] .= β2 .* cLayer.S[:db] .+ (1-β2) .* (cLayer.dB.^2)

           ##correcting
           SCorrected[:dw] .= cLayer.S[:dw] ./ (1-β2^tMiniBatch)
           SCorrected[:db] .= cLayer.S[:db] ./ (1-β2^tMiniBatch)

           ##update parameters with adam
           cLayer.W .-= (α .* (VCorrected[:dw] ./ (sqrt.(SCorrected[:dw]) .+ ϵAdam)))
           cLayer.B .-= (α .* (VCorrected[:db] ./ (sqrt.(SCorrected[:db]) .+ ϵAdam)))

           VCorrected = nothing
           SCorrected = nothing
       else#if optimizer==:momentum

           cLayer.W .-= (α .* VCorrected[:dw])
           cLayer.B .-= (α .* VCorrected[:db])
           VCorrected = nothing
       end #if optimizer==:adam
   else
       cLayer.W .-= (α .* cLayer.dW)
       cLayer.B .-= (α .* cLayer.dB)
   end #if optimizer==:adam || optimizer==:momentum

   cLayer.updateCount += 1

   # Base.GC.gc()

   return nothing
end #layerUpdateParams!

function layerUpdateParams!(
                      model::Model,
                      cLayer::PL,
                      cnt::Integer = -1;
                      tMiniBatch::Integer = 1,
                      ) where {PL <: PoolLayer}


   if cLayer.updateCount >= cnt
       return nothing
   end #if cLayer.updateCount >= cnt

   if !all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
       return nothing
   end

   cLayer.updateCount += 1

   return nothing
end #updateParams!
