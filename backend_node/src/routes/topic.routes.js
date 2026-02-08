import { Router } from "express";
import {topicselection} from "../controllers/topic.controller.js";

const router = Router()
router.route("/topicselection").post(topicselection)

export default router