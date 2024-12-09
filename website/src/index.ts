import express from 'express'
import { join } from 'path'
import ejs from 'ejs'

const app = express();
app.engine("'ejs", ejs.renderFile)
app.set('views', join(process.cwd(), "static"));
app.use('/', express.static(join(process.cwd(), "static")))
app.use('/api', (request, response, next) => {
    return response.render("index.ejs", { num: 23 })
})
app.listen(80, () => {
    console.debug({
        listening_port: 80,
        static_directory: join(process.cwd(), "static")
    })
})