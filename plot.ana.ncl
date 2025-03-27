begin


    date_tag = "2020-01-07"
    file_prefix = "../backup/" + date_tag

    file_prefix = "."

    ; read ease grid.
    f = addfile("./data/ease2_nh25km_grid.nc", "r")
    lat2d = f->lat
    lon2d = f->lon
    

    ; 1. read sis2 grid
    fg = addfile("./data/sea_ice_geometry.nc", "r")
    geolat = fg->geolat
    geolon = fg->geolon
    wet_mask = fg->wet


    ; 2. read ice model org
    f = addfile(file_prefix + "/ice_model.org.nc", "r")
    p1 = f->part_size(0,:,:,:)
    s1 = 1.0 - p1(0,:,:)

    s1@lat2d = geolat
    s1@lon2d = geolon
    s1 = mask(s1, wet_mask, 1)

    q1 = f->h_ice(0,:,:,:)
    h1 = dim_sum_n(p1(1:5,:,:) * q1, 0) / 903.0

    h1@lat2d = geolat
    h1@lon2d = geolon
    
    h1 = mask(h1, wet_mask, 1)



    ; 3. read ana
    f = addfile(file_prefix + "/ice.ana.nc", "r")
    p2 = f->part_size(0,:,:,:)
    s2 = 1.0 - p2(0,:,:)


    s2@lat2d = geolat
    s2@lon2d = geolon

    s2 = mask(s2, wet_mask, 1)


    q2 = f->h_ice(0,:,:,:)
    h2 = dim_sum_n(p2(1:5,:,:) * q2, 0) / 903.0


    h2@lat2d = geolat
    h2@lon2d = geolon

    h2 = mask(h2, wet_mask, 1)


    ; 4. read cs2smos
    f = addfile(file_prefix + "/obs.nc", "r")
    obs_c = f->obs(0,:,:,0) 
    obs_h = f->obs(0,:,:,1) * 6.8

    obs_c@lat2d = lat2d
    obs_c@lon2d = lon2d 
    obs_c@_FillValue = -999
    obs_c = mask(obs_c, obs_c .gt. 0.1, True)


    obs_h@lat2d = lat2d
    obs_h@lon2d = lon2d
    obs_h@_FillValue = -999
    obs_h = mask(obs_h, obs_h .gt. 0.2, True)


    ; plot
    wks = gsn_open_wks("x11", "plot")
    gsn_define_colormap(wks, "ncl_default")


    res = True
    res@gsnDraw = False
    res@gsnFrame = False
    res@gsnPolar = "NH"
    res@mpMinLatF = 45.
    res@cnLinesOn = False
    res@cnFillOn = True
    res@cnLineLabelsOn = False
    res@gsnAddCyclic = True

    ; res@cnFillMode = "RasterFill"

    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = 0.05
    res@cnMaxLevelValF = 0.95
    res@cnLevelSpacingF = 0.05


    plots = new(8, graphic)


    res@gsnCenterString = "sis2"
    plots(0) = gsn_csm_contour_map_polar(wks, s1, res)

    res@gsnCenterString = "cs2smos"
    plots(1) = gsn_csm_contour_map_polar(wks, obs_c, res)

    res@gsnCenterString = "vae-lspf"
    plots(2) = gsn_csm_contour_map_polar(wks, s2, res)


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = -0.1
    res@cnMaxLevelValF =  0.1
    res@cnLevelSpacingF = 0.01

    s2 = s2 - s1
    s2 = mask(s2, abs(s2) .gt. 1.e-3, True)

    res@gsnCenterString = "increment"
    plots(3) = gsn_csm_contour_map_polar(wks, s2, res)
    
 

    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = 0.2
    res@cnMaxLevelValF = 3.0
    res@cnLevelSpacingF = 0.2

    res@gsnCenterString = "sis2"
    plots(4) = gsn_csm_contour_map_polar(wks, h1, res)
    
    res@gsnCenterString = "cs2smos"
    plots(5) = gsn_csm_contour_map_polar(wks, obs_h, res)


    res@gsnCenterString = "vae-lspf"
    plots(6) = gsn_csm_contour_map_polar(wks, h2, res)


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF = -0.3
    res@cnMaxLevelValF =  0.3
    res@cnLevelSpacingF = 0.03

    h2 = h2 - h1
    h2 = mask(h2, abs(h2) .gt. 1.e-3, True)

    res@gsnCenterString = "increment"
    plots(7) = gsn_csm_contour_map_polar(wks, h2, res)
    

    resp = True

    resp@gsnFrame = False

    gsn_panel(wks, plots, (/2,4/), resp)

    frame(wks)

end