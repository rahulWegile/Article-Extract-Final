import { AppBar, Toolbar, Typography, Button } from "@mui/material";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h5"
          sx={{
            fontWeight: "bold",
            flexGrow: 1,
          }}
        >
          Newspaper Boundary Detector
        </Typography>
        <Button
          component={Link}
          to="/search"
          variant="contained"
          color="secondary"
        >
          View Articles
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;