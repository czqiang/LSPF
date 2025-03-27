    begin

    sit_factor = 10.

    ; 1. read in ease2 grid.
    f = addfile("./data/ease2_nh25km_grid.nc", "r")
    ease_lat = f->lat
    ease_lon = f->lon


    ; 2. read in vae data
    f = addfile("vae.new.nc", "r")  ; vae.new
    sic = f->fused_ice(:,:,0)
    sit = f->fused_ice(:,:,1) * sit_factor
    
    delete(sic@_FillValue)
    delete(sit@_FillValue)


    ; 3. read in sis2 grid
    f = addfile("./data/sea_ice_geometry.nc", "r")
    geolat = f->geolat
    geolon = f->geolon


    ; 4.regrid
    opt = True

    opt@RemoveSrcFile = True
    opt@RemoveDstFile = True
    opt@RemoveWgtFile = True
    
    opt@NoPETLog = True
    opt@RemovePETLog = True
    
    opt@SrcGridLat = ease_lat
    opt@SrcGridLon = ease_lon

    opt@DstGridLat = geolat
    opt@DstGridLon = geolon
    
    opt@SrcRegional = True
    opt@DstRegional = False
    
    opt@ReturnDouble = True

    sic_regrid = (/ESMF_regrid(sic, opt)/)
    sit_regrid = (/ESMF_regrid(sit, opt)/)


    ; in (0,1)
    sic_regrid = where(sic_regrid .lt. 0., 0., sic_regrid)
    sic_regrid = where(sic_regrid .gt. 1., 1., sic_regrid)
    sit_regrid = where(sit_regrid .lt. 0., 0., sit_regrid)

    ; save
    f = addfile("ice.vae.nc", "c")
    f->sic = (/sic_regrid/)
    f->sit = (/sit_regrid/)

end