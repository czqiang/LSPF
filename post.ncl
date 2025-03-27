begin

    ; read in obs sea ice
    f = addfile("ice.vae.nc", "r")
    sic_obs = f->sic
    sit_obs = f->sit


    ; read in sea ice info
    f = addfile("./data/sea_ice_geometry.nc", "r")
    geolat = f->geolat
    wet_mask = f->wet

    ; read in sea ice model org
    f  = addfile("ice_model.org.nc", "r")
    ps = f->part_size
    hi = f->h_ice / 903.
    ; hs = f->h_snow
    ; ei = f->enth_ice
    ; es = f->enth_snow

    
    ;; 
    do i = 0, 319
    do j = 0, 359

        if (geolat(i,j) .lt. 40) then
            continue
        end if

        if (wet_mask(i,j) .eq. 0) then
            continue
        end if

        if (ismissing(sic_obs(i,j))) then
            continue
        end if

        if (ismissing(sit_obs(i,j))) then
            continue
        end if

        cell_sic = 1.0 - ps(0,0,i,j)

        if (cell_sic .lt. 1.e-2) then
            continue
        end if

        if (sic_obs(i,j) .lt. 1.e-2) then
            continue
        end if


        ; now sic da
        sic = cell_sic*0.1 + sic_obs(i,j)*0.9 ; avoid large model shock

        sic_ratio = sic / cell_sic

        ; consider use this ratio to scale 
        do k = 1, 5
            if (ps(0,k,i,j) .gt. 1.e-3) then
                ps(0,k,i,j) = ps(0,k,i,j) * sic_ratio
            end if
        end do
        
        ; adjust open water
        new_sic = sum(ps(0,1:5,i,j))
        if (new_sic .gt. 1.000001) then
            print("error: new ice concentration over 1.0!")
            print(new_sic)
            exit
            ; new_sic = 1.0
        end if
        ps(0,0,i,j) = max((/0, 1.0 - new_sic/))


        
        ; consider sit
        bins_sit = ps(0,1:5,i,j) * hi(0,:,i,j)
        cell_sit = sum(bins_sit)


        if (cell_sit .lt. 1.e-2) then
            continue
        end if

        if (sit_obs(i,j) .lt. 1.e-2) then
            continue
        end if


        ; now sit da
        sit = cell_sit*0.1 + sit_obs(i,j)*0.9 ; aviod large model shock

        sit_ratio = sit / cell_sit

        do k = 0, 4
            if (bins_sit(k) .gt. 1.e-3) then
                hi(0,k,i,j) = bins_sit(k) * sit_ratio / ps(0,k+1,i,j) * 903.
            end if
        end do

    end do
    end do

    ; save file 
    fo = addfile("ice.ana.nc", "c")
    fo->part_size = ps
    fo->h_ice = hi
    ; fo->h_snow = hs
    ; fo->enth_ice = ei
    ; fo->enth_snow = es

end