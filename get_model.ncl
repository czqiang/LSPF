begin

    this_name = "new07"
    file_out = "sis2." + this_name + ".nc"

    ; read sis2
    fils = systemfunc("ls ~/2020/glor/" + this_name + "/HISTORY/ice_2020_* | sort -n | head -n 75")
    fs   = addfiles(fils, "r")
    cn   = fs[:]->CN
    sit  = fs[:]->HI


    ;; sic
    sic = dim_sum_n_Wrap(cn, 1)
    ; sic_mean = dim_avg_n_Wrap(sic, 0)

    sic@_FillValue = -999.
    ; sic = mask(sic, sic .gt. 0.15, True)



    ;; sit
    sit@_FillValue = -999.
    ; sit_mean = dim_avg_n_Wrap(sit, 0)
    ; sit = mask(sit, sic .gt. 0.15, True)




    ; read geo info
    f = addfile("../data/sea_ice_geometry.nc", "r")
    wet_mask = f->wet
    geolat = f->geolat 
    geolon = f->geolon





    ;; 
    f = addfile("../data/ease2_nh25km_grid.nc", "r")
    ease_lat = f->lat
    ease_lon = f->lon



    ;; regrid
    opt              = True

    ; opt@RemoveSrcFile = True
    ; opt@RemoveDstFile = True
    ; opt@RemoveWgtFile = True

    opt@SkipSrcGrid = True
    opt@SkipDstGrid = True
    opt@SkipWgtGen = True

    opt@NoPETLog     = True
    opt@RemovePETLog = True

    opt@SrcGridLat = geolat
    opt@SrcGridLon = geolon

    opt@DstGridLat   = ease_lat
    opt@DstGridLon   = ease_lon

    opt@SrcRegional  = False
    opt@DstRegional  = True

    ;;
    ; opt@SrcGridMask  = where(.not.ismissing(sic),1,0)
    opt@SrcGridMask  = wet_mask
    sic_regrid       = ESMF_regrid(sic, opt)

    printVarSummary(sic_regrid)

    ;;
    ; opt@SrcGridMask  = where(.not.ismissing(sit),1,0)
    sit_regrid       = ESMF_regrid(sit, opt)

    printVarSummary(sit_regrid)

    sic_regrid@_FillValue = -999.
    sit_regrid@_FillValue = -999.


    fo = addfile(file_out, "c")
    fo->sic = (/sic_regrid/)
    fo->sit = (/sit_regrid/)



    ; wks = gsn_open_wks("x11", "x")
    ; gsn_define_colormap(wks, "cmp_b2r")

    ; res = True
    ; res@gsnPolar = "NH"
    ; res@mpMinLatF = 40.
    ; res@cnLinesOn = False
    ; res@cnFillOn = True

    ; plot = gsn_csm_contour_map_polar(wks, sit_regrid, res)

end