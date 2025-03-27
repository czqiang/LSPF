begin


    nnn = "070"
    vae_nn = "new07"

    ; read ease grid.
    f = addfile("./data/ease2_nh25km_grid.nc", "r")
    lat2d = f->lat
    lon2d = f->lon

    ; 1. read sis2 grid
    fg = addfile("./data/sea_ice_geometry.nc", "r")
    geolat = fg->geolat
    geolon = fg->geolon


    ; 2. read ctl
    f = addfile("/home/xeon/2020/glor/ctl/HISTORY/ice_2020_" + nnn + ".nc" , "r")
    cn1 = f->CN(0,:,:,:)
    s1 = dim_sum_n(cn1, 0)

    s1@lat2d = geolat
    s1@lon2d = geolon
    s1 = mask(s1, s1 .gt. 0.15, 1)

    h1 = (/f->HI(0,:,:)/)
    h1@lat2d = geolat
    h1@lon2d = geolon
    h1 = mask(h1, s1 .gt. 0.15, 1)


    ; 3. read vae
    f = addfile("/home/xeon/2020/glor/" + vae_nn + "/HISTORY/ice_2020_" + nnn + ".nc", "r")
    cn2 = f->CN(0,:,:,:)
    s2 = dim_sum_n(cn2, 0)

    s2@lat2d = geolat
    s2@lon2d = geolon
    s2 = mask(s2, s2 .gt. 0.15, 1)

    h2 = (/f->HI(0,:,:)/)
    h2@lat2d = geolat
    h2@lon2d = geolon
    h2 = mask(h2, s2 .gt. 0.15, 1)



    ; 4. read cs2smos
    ; f = addfile("/home/xeon/2020/fix/cs2smos/dst/f" + nnn + ".nc", "r")
    f = addfile("/home/xeon/2020/fix/cs2smos/dst/f106.nc", "r")
    obs_c = f->obs(0,:,:,0) 
    obs_h = f->obs(0,:,:,1) * 10.

    obs_c@lat2d = lat2d
    obs_c@lon2d = lon2d 
    obs_c@_FillValue = -999
    obs_c = mask(obs_c, obs_c .gt. 0.15, True)

    obs_h@lat2d = lat2d
    obs_h@lon2d = lon2d
    obs_h@_FillValue = -999
    obs_h = mask(obs_h, obs_c .gt. 0.15, True)



    ; plot
    wks = gsn_open_wks("x11", "plot")
    gsn_define_colormap(wks, "ncl_default")


    res = True
    res@gsnDraw = False
    res@gsnFrame = False
    res@gsnPolar = "NH"
    res@mpMinLatF = 55.
    res@cnLinesOn = False
    res@cnFillOn = True
    res@cnLineLabelsOn = False
    res@gsnAddCyclic = True

    ; res@cnFillMode = "RasterFill"

    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = 0.2
    res@cnMaxLevelValF = 0.9
    res@cnLevelSpacingF = 0.1


    plots = new(8, graphic)


    res@gsnCenterString = "sis2"
    plots(0) = gsn_csm_contour_map_polar(wks, s1, res)

    res@gsnCenterString = "cs2smos"
    plots(1) = gsn_csm_contour_map_polar(wks, obs_c, res)

    res@gsnCenterString = "vae-lspf"
    plots(2) = gsn_csm_contour_map_polar(wks, s2, res)


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = -0.12
    res@cnMaxLevelValF =  0.12
    res@cnLevelSpacingF = 0.02

    s2 = s2 - s1
    s2 = mask(s2, abs(s2) .gt. 1.e-3, True)

    res@gsnCenterString = "increment"
    plots(3) = gsn_csm_contour_map_polar(wks, s2, res)
    
 


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = 0.2
    res@cnMaxLevelValF = 3.4
    res@cnLevelSpacingF = 0.2

    res@gsnCenterString = "sis2"
    plots(4) = gsn_csm_contour_map_polar(wks, h1, res)
    
    res@gsnCenterString = "cs2smos"
    plots(5) = gsn_csm_contour_map_polar(wks, obs_h, res)


    res@gsnCenterString = "vae-lspf"
    plots(6) = gsn_csm_contour_map_polar(wks, h2, res)


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = -1.
    res@cnMaxLevelValF =  1.
    res@cnLevelSpacingF = 0.1

    h2 = h2 - h1
    h2 = mask(h2, abs(h2) .gt. 1.e-3, True)

    res@gsnCenterString = "increment"
    plots(7) = gsn_csm_contour_map_polar(wks, h2, res)
    




    resp = True

    resp@gsnFrame = False

    gsn_panel(wks, plots, (/2,4/), resp)

    frame(wks)

end