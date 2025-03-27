begin

    ; read geo info
    f = addfile("~/ice_data/noaa/G02202-cdr-ancillary-nh.nc", "r")
    latitude = f->latitude
    longitude = f->longitude
    landmask = byte2flt(f->landmask)


    ; read data
    fils = systemfunc("ls ~/ice_data/noaa/src/* | sort -n | head -n 106 | tail -n 75")
    fs = addfiles(fils, "r")
    sic = byte2flt(fs[:]->cdr_seaice_conc)

    sic_mean = dim_avg_n_Wrap(sic, 0)



    ;;
    sic_mean@lat2d = latitude
    sic_mean@lon2d = longitude

    sic_mean = mask(sic_mean, landmask .eq. 0, True)
    sic_mean = mask(sic_mean, sic_mean .gt. 0.15, True)
    
    printMinMax(sic_mean, False)


    ;; 
    f = addfile("../data/ease2_nh25km_grid.nc", "r")
    ease_lat = f->lat
    ease_lon = f->lon



    ;; regrid
    opt              = True

    opt@RemoveSrcFile = True
    opt@RemoveDstFile = True
    opt@RemoveWgtFile = True

    opt@NoPETLog     = True
    opt@RemovePETLog = True

    opt@DstGridLat   = ease_lat
    opt@DstGridLon   = ease_lon

    opt@SrcRegional  = True
    opt@DstRegional  = True

    opt@SrcGridMask    = where(.not.ismissing(sic_mean),1,0)

    sic_regrid       = ESMF_regrid(sic_mean, opt)


    ; fo = addfile("noaa_cdr.sic.nc", "c")

    ; sic_regrid@_FillValue = -999.
    ; fo->sic_mean = (/sic_regrid/)
    



    wks = gsn_open_wks("x11", "x")
    gsn_define_colormap(wks, "cmp_b2r")

    res = True
    res@gsnPolar = "NH"
    res@mpMinLatF = 40.
    res@cnLinesOn = False
    res@cnFillOn = True

    plot = gsn_csm_contour_map_polar(wks, sic_regrid, res)


end