const express = require('express')
const app = express()
const port = process.env.PORT || 3000

const fileUpload = require('express-fileupload');
app.use(fileUpload());

const pyCmd = require("node-cmd");
const PS = require('python-shell');

app.get('/', (req, res) => 
{
    res.sendFile(__dirname + "/index.html");
})

app.post("/upload", (req, res) =>
{
    console.log(req.files);

    if(req.files)
    {
        let file = req.files.dataset;
        let filename = file.name;

        // console.log(Buffer.from(file.data).toString("base64"));

        var pyshell = new PS.PythonShell('cursi.py');
        pyshell.send(Buffer.from(file.data).toString("base64"));

        pyshell.stdout.on('data', function(data) 
        {
            console.log(data.toString());
            res.write(data);
            res.end();
        });

        // // end the input stream and allow the process to exit
        // pyshell.end(function (err,code,signal) 
        // {
        //     if (err) throw err;
        //     console.log('The exit code was: ' + code);
        //     console.log('The exit signal was: ' + signal);
        //     console.log('finished');
        // });

        // file.mv(`${__dirname}/upload/${filename}`, (err) =>
        // {
        //     if(err)
        //     {
        //         console.log(err);
        //         res.send("Eroare boss");
        //     }
        //     else
        //     {
        //         pyCmd.run(`python dummy.py ${__dirname}/upload/${filename}`).stdout.on('data', function(data) 
        //         {
        //             console.log(data.toString());
        //             res.write(data);
        //             res.end();
        //         });
        //     }
        // });
    }
});

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`);
})