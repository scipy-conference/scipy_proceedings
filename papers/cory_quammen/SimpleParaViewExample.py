from paraview import servermanager as sm

pm = sm.vtkSMProxyManager.GetProxyManager()
controller = \
    sm.vtkSMParaViewPipelineControllerWithRendering()

ss = pm.NewProxy('sources', 'SphereSource')
ss.GetProperty('Radius').SetElement(0, 2.0)
controller.RegisterPipelineProxy(ss)

view = pm.GetProxy('views', 'RenderView1')
rep = view.CreateDefaultRepresentation(ss, 0)
controller.RegisterRepresentationProxy(rep)
rep.GetProperty('Input').SetInputConnection(0, ss, 0)
rep.GetProperty('Visibility').SetElement(0, 1)

controller.Show(ss, 0, view)
view.ResetCamera()
view.StillRender()
