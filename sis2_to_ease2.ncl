begin

    sit_factor = 10.

    ; 1. read sis2
    f = addfile("./data/sea_ice_geometry.nc", "r")
    geolat = doubletofloat(f->geolat)
    geolon = doubletofloat(f->geolon)


    ; 2. read sis2 model ice
    f = addfile("ice_model.org.nc", "r")
    ps = doubletofloat(f->part_size(0,:,:,:))
    hi = doubletofloat(f->h_ice(0,:,:,:)) / 903.0

    sic = 1.0 - ps(0,:,:)
    sit = dim_sum_n(ps(1:5,:,:)*hi, 0)
    

    ; 3. read ease2 grid.
    f = addfile("./data/ease2_nh25km_grid.nc", "r")
    ease_lat = f->lat
    ease_lon = f->lon


    ; 4. opt
    opt = True

    opt@RemoveSrcFile = True
    opt@RemoveDstFile = True
    opt@RemoveWgtFile = True

    opt@NoPETLog = True
    opt@RemovePETLog = True
    
    opt@SrcGridLat = geolat
    opt@SrcGridLon = geolon

    opt@DstGridLat = ease_lat
    opt@DstGridLon = ease_lon
    
    opt@SrcRegional = False
    opt@DstRegional = True


    ; regrid
    ice = new((/1, 432, 432, 2/), "float", 0.)
    
    sic_reg = (/ESMF_regrid(sic, opt)/)
    sit_reg = (/ESMF_regrid(sit, opt)/)


    ; in (0,1)
    sic_reg = where(sic_reg .lt. 0., 0., sic_reg)
    sic_reg = where(sic_reg .gt. 1., 1., sic_reg)

    sit_reg = where(sit_reg .lt. 0., 0., sit_reg)
    sit_reg = where(sit_reg .gt. sit_factor, sit_factor, sit_reg)
    
    
    ; norm sit
    sit_reg = sit_reg / sit_factor
    
    
    ; insert 
    ice(0,:,:,0) = (/sic_reg/)
    ice(0,:,:,1) = (/sit_reg/)

    ; info me
    print("")
    print("sis2 -> ease2")
    print("norm sic: ") 
    printMinMax(sic_reg, False)
    print("norm sit: ") 
    printMinMax(sit_reg, False)
    print("")
    
    ; save
    delete(ice@_FillValue) 
    
    fo = addfile("sis2.nc", "c")
    fo->sis2 = ice

end