begin

    min_sic = 0.15

    ; cs2smos
    f                  = addfile("cs2.nc", "r")
    cs2_sic            = f->sic
    cs2_sic@_FillValue = -999.
    cs2_sic = mask(cs2_sic, cs2_sic .gt. min_sic, True)
    

    cs2_sit            = f->sit
    cs2_sit@_FillValue = -999.
    cs2_sit = mask(cs2_sit, cs2_sic .gt. min_sic, True)



    ; ctl
    f                  = addfile("sis2.ctl.nc", "r")
    ctl_sic            = f->sic
    ctl_sic@_FillValue = -999.
    ctl_sic = mask(ctl_sic, ctl_sic .gt. min_sic, True)

    
    ctl_sit            = f->sit
    ctl_sit@_FillValue = -999.
    ctl_sit = mask(ctl_sit, ctl_sic .gt. min_sic, True)



    ; new-03
    f                  = addfile("sis2.new03.nc", "r")
    n03_sic            = f->sic
    n03_sic@_FillValue = -999.
    n03_sic = mask(n03_sic, n03_sic .gt. min_sic, True)


    n03_sit            = f->sit
    n03_sit@_FillValue = -999.
    n03_sit = mask(n03_sit, n03_sic .gt. min_sic, True)



    ; mew-07
    f                  = addfile("sis2.new07.nc", "r")
    n07_sic            = f->sic
    n07_sic@_FillValue = -999.
    n07_sic = mask(n07_sic, n07_sic .gt. min_sic, True)


    n07_sit            = f->sit
    n07_sit@_FillValue = -999.
    n07_sit = mask(n07_sit, n07_sic .gt. min_sic, True)



    ;;
    f               = addfile("../data/ease2_nh25km_grid.nc", "r")
    lat             = f->lat
    lon             = f->lon
    


    ;; rmsd
    rmse_ctl_sic = dim_rmsd_n_Wrap(cs2_sic, ctl_sic, 0)
    ; rmse_ctl_sic = mask(rmse_ctl_sic, rmse_ctl_sic .lt. 0.8, True)


    rmse_ctl_sit = dim_rmsd_n_Wrap(cs2_sit, ctl_sit, 0)

    
    rmse_n03_sic = dim_rmsd_n_Wrap(cs2_sic, n03_sic, 0)
    ; rmse_n03_sic = mask(rmse_n03_sic, rmse_n03_sic .lt. 0.8, True)


    rmse_n03_sit = dim_rmsd_n_Wrap(cs2_sit, n03_sit, 0)

    
    
    rmse_n07_sic = dim_rmsd_n_Wrap(cs2_sic, n07_sic, 0)
    ; rmse_n07_sic = mask(rmse_n07_sic, rmse_n07_sic .lt. 0.8, True)

    
    rmse_n07_sit = dim_rmsd_n_Wrap(cs2_sit, n07_sit, 0)




    ;;
    rmse_ctl_sic@lat2d      = lat
    rmse_ctl_sic@lon2d      = lon

    rmse_ctl_sit@lat2d      = lat
    rmse_ctl_sit@lon2d      = lon


    
    rmse_n03_sic@lat2d      = lat
    rmse_n03_sic@lon2d      = lon

    rmse_n03_sit@lat2d      = lat
    rmse_n03_sit@lon2d      = lon



    rmse_n07_sic@lat2d      = lat
    rmse_n07_sic@lon2d      = lon

    rmse_n07_sit@lat2d      = lat
    rmse_n07_sit@lon2d      = lon




    ;;
    wks = gsn_open_wks("pdf", "figure4")

    res                      = True
    res@gsnFrame             = False
    res@gsnDraw              = False
    res@gsnPolar             = "NH"
    res@mpMinLatF            = 55.
    res@cnLinesOn            = False
    res@cnFillOn             = True

    res@lbLabelFontHeightF = 0.022

    plots                    = new(6, graphic)

    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF       = 0.02
    res@cnMaxLevelValF       = 0.4
    res@cnLevelSpacingF      = 0.02

    res@cnFillPalette        = "WhBlGrYeRe"

    res@gsnStringFontHeightF = 0.02  

    res@gsnLeftString = "(a)"
    res@gsnCenterString = "SIC RMSE: CTL (DA Period)"
    res@gsnRightString = "[0-1]"
    plots(0)                 = gsn_csm_contour_map_polar(wks, rmse_ctl_sic, res)

    ; res@cnFillPalette        = "BlueWhiteOrangeRed"

    ; res@cnLevelSelectionMode = "ManualLevels"
    ; res@cnMinLevelValF       = -0.08
    ; res@cnMaxLevelValF       =  0.08
    ; res@cnLevelSpacingF      =  0.01

    res@gsnLeftString = "(b)"
    res@gsnCenterString = "SIC RMSE: LSPF-07 (DA Period)"
    res@gsnRightString = "[0-1]"
    plots(1)                 = gsn_csm_contour_map_polar(wks, rmse_n07_sic, res)

    res@gsnLeftString = "(c)"
    res@gsnCenterString = "SIC RMSE: LSPF-03 (DA Period)"
    res@gsnRightString = "[0-1]"
    plots(2)                 = gsn_csm_contour_map_polar(wks, rmse_n03_sic, res)


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF       =  0.05
    res@cnMaxLevelValF       =  1.25
    res@cnLevelSpacingF      =  0.05


    res@gsnLeftString = "(d)"
    res@gsnCenterString = "SIT RMSE: CTL (DA Period)"
    res@gsnRightString = "[m]"
    plots(3)                 = gsn_csm_contour_map_polar(wks, rmse_ctl_sit, res)


    ; res@cnLevelSelectionMode = "ManualLevels"
    ; res@cnMinLevelValF       = 0.2
    ; res@cnMaxLevelValF       = 3.4
    ; res@cnLevelSpacingF      = 0.2

    ; res@cnFillPalette        = "cmp_b2r"

    res@gsnLeftString = "(e)"
    res@gsnCenterString = "SIT RMSE: LSPF-07 (DA Period)"
    res@gsnRightString = "[m]"
    plots(4)                 = gsn_csm_contour_map_polar(wks, rmse_n07_sit, res)

    ; res@cnFillPalette        = "BlueWhiteOrangeRed"

    ; res@cnLevelSelectionMode = "ManualLevels"
    ; res@cnMinLevelValF       = -1.4
    ; res@cnMaxLevelValF       =  1.4
    ; res@cnLevelSpacingF      =  0.1

    res@gsnLeftString = "(f)"
    res@gsnCenterString = "SIT RMSE: LSPF-03 (DA Period)"
    res@gsnRightString = "[m]"
    plots(5)                 = gsn_csm_contour_map_polar(wks, rmse_n03_sit, res)

    ; res@gsnLeftString = "(g)"
    ; res@gsnCenterString = "DIFF: LSPF-07 - CS2SMOS"
    ; res@gsnRightString = "[m]"
    ; plots(6)                 = gsn_csm_contour_map_polar(wks, dif2_sit, res)

    ; res@gsnLeftString = "(h)"
    ; res@gsnCenterString = "DIFF: LSPF-03 - CS2SMOS"
    ; res@gsnRightString = "[m]"
    ; plots(7)                 = gsn_csm_contour_map_polar(wks, dif3_sit, res)



    resp                 = True
    resp@gsnFrame        = False
    ; resp@gsnPanelRowSpec = True
    resp@gsnPanelYWhiteSpacePercent = 10
    resp@gsnPanelXWhiteSpacePercent = 5

    gsn_panel(wks, plots, (/2,3/), resp)

    frame(wks)

end