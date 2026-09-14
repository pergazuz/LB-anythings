"""Label configs shared by tests. The first is the pipe project's real config."""

PIPE = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pipe" background="#FF0000"/>
  </RectangleLabels>
</View>
"""

VEHICLES = """
<View>
  <Header value="Draw boxes"/>
  <Image name="img" value="$photo" zoom="true"/>
  <RectangleLabels name="boxes" toName="img">
    <Label value="car"/>
    <Label value="truck"/>
  </RectangleLabels>
</View>
"""
