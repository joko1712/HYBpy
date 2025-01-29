import React from "react";
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Grid,
    Typography,
} from "@mui/material";

const SimulationModal = ({ open, handleClose }) => (
    <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Simulation in Progress</DialogTitle>
        <DialogContent>
            <Grid container spacing={2} marginTop={2}>
                <Grid item xs={12}>
                    <Typography id='modal-modal-description' sx={{ mt: 2 }}>
                        Your simulation has started. You will be redirected to the dashboard once you
                        close this modal.
                    </Typography>
                </Grid>
            </Grid>
        </DialogContent>
        <DialogActions>
            <Button onClick={handleClose} color='primary'>
                Close
            </Button>
        </DialogActions>
    </Dialog>
);

export default SimulationModal;
