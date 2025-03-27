begin

    sit_factor = 10.

    

    ; read ease grid.
    f = addfile("../data/ease2_nh25km_grid.nc", "r")
    lat2d = f->lat
    lon2d = f->lon

    ; read sis2
    f = addfile("../sis2.nc", "r")
    sis2_c = f->sis2(0,:,:,0) 
    sis2_h = f->sis2(0,:,:,1) * sit_factor

    sis2_c@lat2d = lat2d
    sis2_c@lon2d = lon2d
    sis2_c@_FillValue = -999
    sis2_c = mask(sis2_c, sis2_c .gt. 0.15, True)

    sis2_h@lat2d = lat2d
    sis2_h@lon2d = lon2d
    sis2_h@_FillValue = -999
    sis2_h = mask(sis2_h, sis2_c .gt. 0.15, True)


    ; read obs cs2smos
    f = addfile("../obs.nc", "r")
    obs_c = f->obs(0,:,:,0) 
    obs_h = f->obs(0,:,:,1) * sit_factor

    obs_c@lat2d = lat2d
    obs_c@lon2d = lon2d 
    obs_c@_FillValue = -999
    obs_c = mask(obs_c, obs_c .gt. 0.15, True)

    obs_h@lat2d = lat2d
    obs_h@lon2d = lon2d
    obs_h@_FillValue = -999
    obs_h = mask(obs_h, obs_c .gt. 0.15, True)


    ; read vae
    f = addfile("../vae.new.nc", "r")
    vae_c = f->fused_ice(:,:,0) 
    vae_h = f->fused_ice(:,:,1) * sit_factor

    vae_c@lat2d = lat2d
    vae_c@lon2d = lon2d
    vae_c@_FillValue = -999
    vae_c = mask(vae_c, vae_c .gt. 1.e-2, True)

    vae_h@lat2d = lat2d
    vae_h@lon2d = lon2d
    vae_h@_FillValue = -999
    vae_h = mask(vae_h, vae_c .gt. 1.e-2, True)


    ; plot
    wks = gsn_open_wks("x11", "plot")
    gsn_define_colormap(wks, "ncl_default")

    res = True
    res@gsnDraw = False
    res@gsnFrame = False
    res@gsnPolar = "NH"
    res@mpMinLatF = 45.0
    res@cnFillOn = True
    res@cnLinesOn = False
    res@cnFillMode = "RasterFill"


    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF  = 0.05
    res@cnMaxLevelValF  = 0.95
    res@cnLevelSpacingF = 0.05


    plots = new(6, graphic)

    res@gsnCenterString = "sis2"
    plots(0) = gsn_csm_contour_map_polar(wks, sis2_c, res)

    res@gsnCenterString = "cs2smos"
    plots(1) = gsn_csm_contour_map_polar(wks, obs_c, res)

    res@gsnCenterString = "vae"
    plots(2) = gsn_csm_contour_map_polar(wks, vae_c, res)




    res@cnLevelSelectionMode = "ManualLevels"
    res@cnMinLevelValF  = 0.2
    res@cnMaxLevelValF  = 4.2
    res@cnLevelSpacingF = 0.2


    res@gsnCenterString = "sis2"
    plots(3) = gsn_csm_contour_map_polar(wks, sis2_h, res)

    res@gsnCenterString = "cs2smos"
    plots(4) = gsn_csm_contour_map_polar(wks, obs_h, res)
    
    res@gsnCenterString = "vae-pf"
    plots(5) = gsn_csm_contour_map_polar(wks, vae_h, res)



    resp = True
    resp@gsnFrame = False

    gsn_panel(wks, plots, (/2,3/), resp)

    frame(wks)

end