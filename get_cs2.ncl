begin

    fils = systemfunc("ls ../../fix/cs2smos/dst/* | sort -n | tail -n 75")

    fs = addfiles(fils, "r")
    sic = fs[:]->obs(:,:,:,0)
    sit = fs[:]->obs(:,:,:,1) * 10.

    ; printVarSummary(sic)


    ; sic_mean = dim_avg_n(sic, 0)
    ; sit_mean = dim_avg_n(sit, 0)

    sic@_FillValue = -999.
    ; sic = mask(sic, sic .gt. 0.15, True)

    
    sit@_FillValue = -999.
    ; sit = mask(sit, sic .gt. 0.15, True)

   
    ; f = addfile("../data/ease2_nh25km_grid.nc", "r")
    ; sit_mean@lat2d = f->lat
    ; sit_mean@lon2d = f->lon


    fo = addfile("cs2.nc", "c")
    
    sic@_FillValue = -999.
    fo->sic = (/sic/)

    sit@_FillValue = -999.
    fo->sit = (/sit/)


    ; printVarSummary(sit_mean)


    ; wks = gsn_open_wks("x11", "x")
    ; gsn_define_colormap(wks, "cmp_b2r")

    ; res = True
    ; res@gsnPolar = "NH"
    ; res@mpMinLatF = 40.
    ; res@cnLinesOn = False
    ; res@cnFillOn = True

    ; plot = gsn_csm_contour_map_polar(wks, sit_mean, res)

end