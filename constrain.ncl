begin

    sit_factor = 10.

    ; read sis2
    f = addfile("sis2.nc", "r")
    sis2_cn = f->sis2(0,:,:,0) 
    sis2_hi = f->sis2(0,:,:,1) * sit_factor

    ; read obs cs2smos
    f = addfile("obs.nc", "r")
    obs_cn = f->obs(0,:,:,0) 
    obs_hi = f->obs(0,:,:,1) * sit_factor

    ; read vae
    f = addfile("vae.nc", "r")
    fused_ice = (/f->fused_ice/)
    vae_cn = fused_ice(:,:,0) 
    vae_hi = fused_ice(:,:,1) * sit_factor

    ;; 
    total_cn = 0
    pos_cn = 0
    neg_cn = 0

    total_hi = 0
    pos_hi = 0
    neg_hi = 0


    ;;
    do i = 0, 431
    do j = 0, 431

        ; constrain sic
        if (sis2_cn(i,j) .lt. 1.e-3) then
            continue
        else
            total_cn = total_cn + 1
            
            dif_obs = obs_cn(i,j) - sis2_cn(i,j)
            dif_vae = vae_cn(i,j) - sis2_cn(i,j)

            if ((dif_obs .gt. 0. .and. dif_vae .gt. 0.) .or. (dif_obs .lt. 0. .and. dif_vae .lt. 0.)) then
                
                if (abs(dif_vae) .gt. abs(dif_obs)*0.5) then
                    pos_cn = pos_cn + 1
                    vae_cn(i,j) = sis2_cn(i,j) + dif_obs*0.5
                end if                    
            
            else
                neg_cn = neg_cn + 1
                ; vae_cn(i,j) = sis2_cn(i,j) + dif_obs*0.01
                vae_cn(i,j) = sis2_cn(i,j) + dif_vae*0.5
            
            end if

        end if
           

        ; constrain sit
        if (sis2_hi(i,j) .lt. 1.e-3) then
            continue
        else
            total_hi = total_hi + 1

            dif_obs = obs_hi(i,j) - sis2_hi(i,j)
            dif_vae = vae_hi(i,j) - sis2_hi(i,j)

            if ((dif_obs .gt. 0. .and. dif_vae .gt. 0.) .or. (dif_obs .lt. 0. .and. dif_vae .lt. 0.)) then
                
                if (abs(dif_vae) .gt. abs(dif_obs)*0.5) then
                    pos_hi = pos_hi + 1
                    vae_hi(i,j) = sis2_hi(i,j) + dif_obs*0.5
                end if
            
            else
                neg_hi = neg_hi + 1
                ; vae_hi(i,j) = sis2_hi(i,j) + dif_obs*0.01
                vae_hi(i,j) = sis2_hi(i,j) + dif_vae*0.5
            
            end if
        end if

    end do
    end do

        ; save
        fused_ice(:,:,0) = vae_cn
        fused_ice(:,:,1) = vae_hi / sit_factor

        f = addfile("vae.new.nc", "c")
        f->fused_ice = fused_ice

end