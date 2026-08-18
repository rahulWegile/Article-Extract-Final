import { AppBar, Toolbar, Typography } from "@mui/material";

function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h5"
          sx={{
            fontWeight: "bold",
          }}
        >
          Newspaper Boundary Detector
        </Typography>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;