import wx
import wx.grid as gridlib

class MyFrame(wx.Frame):
    """
    This is MyFrame.  It just shows a few controls on a wxPanel,
    and has a simple menu.
    """

    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, -1, title, size=(1000, 600))

        # Now create the Panel to put the other controls on.
        panel = wx.Panel(self)

        left_panel = wx.Panel(panel, -1, size=(300, 600))
        left_panel.SetBackgroundColour('#6f8089')
        right_panel = wx.Panel(panel, -1)




        left_panel_lower = wx.Panel(left_panel, -1, )
        left_panel_upper = wx.Panel(left_panel, -1)

        leftGridSizer = wx.GridSizer(cols=2, vgap=1, hgap=1)
        left_panel_upper.SetSizer(leftGridSizer)

        add_cell_col_chk = wx.CheckBox(left_panel_lower, label="Add column for cell label using file name",pos=(10, 10))
        choose_files_button = wx.Button(left_panel_lower, label="Choose files", pos=(10, 40))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(left_panel_upper, 1, wx.SHAPED | wx.ALL, border=2)
        sizer.Add(left_panel_lower, 0, wx.EXPAND | wx.ALL, border=2)
        left_panel.SetSizer(sizer)



        mainGrid = gridlib.Grid(right_panel)
        mainGrid.CreateGrid(1, 1)

        # set relative position and add grid to right panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(mainGrid, 1, wx.EXPAND)
        right_panel.SetSizer(sizer)

        # add left and right panels to main panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(left_panel, 0, wx.EXPAND | wx.ALL, border=2)
        sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, border=2)
        panel.SetSizer(sizer)

        self.Show()

        # and a few controls
        # text = wx.StaticText(panel, -1, "Hello World!")
        # text.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
        # text.SetSize(text.GetBestSize())
        # btn = wx.Button(panel, -1, "Close")
        # funbtn = wx.Button(panel, -1, "Just for fun...")

        # bind the button events to handlers
        # self.Bind(wx.EVT_BUTTON, self.OnTimeToClose, btn)
        # self.Bind(wx.EVT_BUTTON, self.OnFunButton, funbtn)

        # Use a sizer to layout the controls, stacked vertically and with
        # a 10 pixel border around each
        # sizer = wx.BoxSizer(wx.VERTICAL)
        # sizer.Add(text, 0, wx.ALL, 10)
        # sizer.Add(btn, 0, wx.ALL, 10)
        # sizer.Add(funbtn, 0, wx.ALL, 10)
        # panel.SetSizer(sizer)
        # panel.Layout()

    def OnTimeToClose(self, evt):
        """Event handler for the button click."""
        print("See ya later!")
        self.Close()

    def OnFunButton(self, evt):
        """Event handler for the button click."""
        print("Having fun yet?")


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(None, "Simple wxPython App")
    app.MainLoop()

# class MyApp(wx.App):
#     def OnInit(self):
#         frame = MyFrame(None, "Simple wxPython App")
#         self.SetTopWindow(frame)
#
#         print("Print statements go to this stdout window by default.")
#
#         frame.Show(True)
#         return True
#
#
# app = MyApp(redirect=True)
# app.MainLoop()