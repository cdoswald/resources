Attribute VB_Name = "Module1"
Const PointsPerInch = 72

Sub CropSelectedImage()

With ActiveWindow.Selection.ShapeRange(1)
    .PictureFormat.Crop.ShapeWidth = 8.65 * PointsPerInch
    .PictureFormat.Crop.ShapeHeight = 6.12 * PointsPerInch
    .PictureFormat.Crop.ShapeLeft = 2.34 * PointsPerInch
    .PictureFormat.Crop.ShapeTop = 0.99 * PointsPerInch
    .PictureFormat.Crop.PictureWidth = 11.64 * PointsPerInch
    .PictureFormat.Crop.PictureHeight = 6.9 * PointsPerInch
    .PictureFormat.Crop.PictureOffsetX = -1.11 * PointsPerInch
    .PictureFormat.Crop.PictureOffsetY = -0.2 * PointsPerInch
    
    .ZOrder msoSendToBack
End With

End Sub
