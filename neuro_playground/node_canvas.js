import { Elemental, passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import { fabric, FabricCanvas } from "./fabric.js"
import { pointsToFunction } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/points_to_function.js"
import { TimelineManager } from "./timeline_manager.js"
import { linearSteps } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/linear_steps.js"
import { zipLong as zip } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/zip_long.js"

function getCenter(node) {
    const { left, top, width, height } = node.getBoundingRect()
    return { left: left + width / 2, top: top + height / 2 }
}
function easeInOutQuad(currentTime, startValue, changeInValue, duration) {
    currentTime /= duration / 2
    if (currentTime < 1) {
        return (changeInValue / 2) * currentTime * currentTime + startValue
    }
    currentTime--
    return (-changeInValue / 2) * (currentTime * (currentTime - 2) - 1) + startValue
}
function duplicateAnimationUpThenDown({start, end, steps}) {
    let values = linearSteps({ start, end, quantity: Math.floor(steps/2),})
    // up then down
    values = values.concat(values.toReversed().slice(1))
    // deal with off by one edgecases
    values.push(start)
    values = values.slice(0, steps)
    return values
}

const hslBlueHue = 213
const hslYellowHue = 49
const hslRedHue = 14
const energyToHueBase = pointsToFunction({
    xValues: [0, 1],
    yValues: [hslBlueHue, hslYellowHue],
    areSorted: true,
    method: "linearInterpolation",
})
const energyToHue = (energy)=>{
    let hue = energyToHueBase(energy)
    // dumb, yes, but JS doesn't have a modulo operator (it has a remainder operator)
    while (hue < 0) {
        hue += 255 // probably only ever does 1 iteration
    }
    return hue % 255 // remainder operator (same as modulo for positive numbers)
}

const allNodes = new Map()
globalThis.allNodes = allNodes // debugging
const getAllNodes = ()=>{
    return Array.from(allNodes.values()).map(each=>each.deref()).filter(each=>each)
}
export class FabricNode extends fabric.Circle {
    static get type() {
        return "FabricNode";
    }
    constructor({ label, left=100, top=100, radius=50, spikeThreshold=1, energy=0.1, energyDecayRate=0.1, id, outputNodeMapping=[], inputNodeIds=[], isFiring=false, stableEnergyLevel=0.1, energyAfterFiring=0, ...custom }) {
        super({
            radius,
            "startAngle": 0,
            "endAngle": 360,
            "counterClockwise": false,
            "type": "Circle",
            "version": "6.6.1",
            left,
            top,
            // "width": 100,
            // "height": 100,
            "fill": `hsl(${energyToHue(energy)}, 100%, 50%)`,
            "stroke": "hsl(0, 0%, 82.75%)",
            "strokeWidth": 5,
            "strokeDashArray": null,
            "strokeLineCap": "butt",
            "strokeDashOffset": 0,
            "strokeLineJoin": "miter",
            "strokeUniform": false,
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": false,
            "flipY": false,
            "opacity": 1,
            "shadow": null,
            "visible": true,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over",
            "skewX": 0,
            "skewY": 0,
            originX: 'center',  // Set origin to center on the X axis
            originY: 'center',  // Set origin to center on the Y axis
            objectCaching: false,
            ...custom,
        })
        this.set('hasControls', false)
        // this.set({
        //     lockScalingX: true,  // Disable horizontal scaling
        //     lockScalingY: true,  // Disable vertical scaling
        // })
        Object.assign(this, {
            id: id||`${Math.random()}`,
            label,
            spikeThreshold,
            energy,
            energyDecayRate,
            outputNodeMapping,
            inputNodeIds,
            isFiring,
            stableEnergyLevel,
            energyAfterFiring,
        })
        allNodes.set(this.id, new WeakRef(this))
        this.updateInputOutputConnections()
    }
    updateInputOutputConnections() {
        // if something is listed as an output, make the other node knows it is an input
        for (const [each, value] of Object.entries(this.outputNodeMapping)) {
            if (allNodes.has(each)) {
                const value = allNodes.get(each).deref()
                if (value) {
                    value.inputNodeIds.includes(this.id) || value.inputNodeIds.push(this.id)
                } else {
                    allNodes.delete(each)
                }
            }
        }
    }
    get outputs() {
        let outputs = []
        for (const [id, value] of Object.entries(this.outputNodeMapping)) {
            if (allNodes.has(id)) {
                const value = allNodes.get(id).deref()
                if (value) {
                    outputs.push(value)
                } else {
                    allNodes.delete(id)
                }
            }
        }
        return outputs
    }
    get inputs() {
        let inputs = [] 
        for (let id of this.inputNodeIds) {
            if (allNodes.has(id)) {
                const value = allNodes.get(id).deref()
                if (value) {
                    inputs.push(value)
                } else {
                    allNodes.delete(id)
                }
            }
        }
        return inputs
    }
    toObject(propertiesToInclude) {
        return {
            ...super.toObject(propertiesToInclude),
            id: this.id,
            label: this.label,
            spikeThreshold: this.spikeThreshold,
            energy: this.energy,
            energyDecayRate: this.energyDecayRate,
            isFiring: this.isFiring,
            outputNodeMapping: this.outputNodeMapping,
            inputNodeIds: this.inputNodeIds,
            stableEnergyLevel: this.stableEnergyLevel,
            energyAfterFiring: this.energyAfterFiring,
        }
    }
}
fabric.classRegistry.setClass(FabricNode)

export class FabricLine extends fabric.Path {
    static get type() {
        return "FabricLine";
    }
    // TODO: somehow clean up lines
    constructor({ startNodeId, endNodeId, ...custom }) {
        let startTop=0, startLeft=0, endTop=0, endLeft=0
        try {
            const { left: startLeft, top: startTop } = getCenter(allNodes.get(startNodeId).deref())
        } catch (error) {
            console.error(`error getting startNodeId center for line with startNodeId ${startNodeId}`, error)
        }
        try {
            const { left: endLeft, top: endTop } = getCenter(allNodes.get(endNodeId).deref())
        } catch (error) {
            console.error(`error getting endNodeId center for line with endNodeId ${endNodeId}`, error)
        }
        let controlTop = (startTop + endTop)/2
        let controlLeft = (startLeft + endLeft)/2
        super(`M ${startX} ${startY} C ${controlX} ${controlY}, ${controlX} ${controlY}, ${endX} ${endY}`, {
            fill: '', 
            stroke: 'gray',
            strokeWidth: 5,
            ...custom,
        })
        // allNodes.set(this.id, new WeakRef(this))
    }
    toObject(propertiesToInclude) {
        return {
            ...super.toObject(propertiesToInclude),
            id: this.id,
            label: this.label,
            spikeThreshold: this.spikeThreshold,
            energy: this.energy,
            energyDecayRate: this.energyDecayRate,
        }
    }
}
fabric.classRegistry.setClass(FabricLine)

// specific for nodes
let prevMouseDownTime
export function NodeCanvas({
    width,
    height,
    backgroundColor,
    selectionColor,
    selectionLineWidth,
    onceFabricLoads,
    inputConnectionStyles={
        stroke: 'blue',
        strokeWidth: 5,
    },
    outputConnectionStyles={
        stroke: 'red',
        strokeWidth: 5,
    },
    jsonObjects,
    // onMouseDown,
    // onObjectAdded,
    // onAfterRender,
    // onMouseMove,
    // onMouseUp,
    // onBeforeSelection,
    // onSelectionCreated,
    // onSelectionCleared,
    // onObjectModified,
    // onObjectSelected,
    // onObjectMoving,
    // onObjectScaling,
    // onObjectRotating,
    // onObjectRemoved,
    ...props
}={}) {
    let element
    
    // 
    // helpers
    // 
        let canvas
        const activeAnimations = new Set()
        const pulse = async (node)=>{
            if (node.stroke.includes('hsl')) {
                if (activeAnimations.has(node.id)) {
                    return
                }
                activeAnimations.add(node.id)
                const eachNode = allNodes.get(node.id).deref()
                if (eachNode) {
                    const halfAnimationDuration = 200
                    const originalColor = node.stroke
                    const originalRadius = node.radius
                    
                    const colorValues = [...node.stroke.match(/\d+/g)].map(each=>each-0)
                    const startHue = colorValues[0]
                    const endHue = hslRedHue
                    const startLightness = colorValues[1]
                    const endLightness = 100
                    const startSaturation = colorValues[2]
                    const endSaturation = 50
                    const startRadius = node.radius
                    const endRadius = node.radius*1.2
                    
                    // 
                    // ramp up
                    //
                    let start = Date.now()
                    let end = start + halfAnimationDuration
                    let now
                    while ((now=Date.now()) < end) {
                        await new Promise((resolve)=>setTimeout(resolve, 0))
                        const newHue = easeInOutQuad(now-start, startHue, endHue-startHue, halfAnimationDuration)
                        const newLightness = easeInOutQuad(now-start, startLightness, endLightness-startLightness, halfAnimationDuration)
                        const newSaturation = easeInOutQuad(now-start, startSaturation, endSaturation-startSaturation, halfAnimationDuration)
                        const newRadius = easeInOutQuad(now-start, startRadius, endRadius-startRadius, halfAnimationDuration)
                        const color = `hsl(${newHue}, ${newLightness}%, ${newSaturation}%)`
                        eachNode.set('stroke', color)
                        eachNode.set('radius', newRadius)
                        canvas.renderAll()
                    }
                    // 
                    // ramp down
                    // 
                    start = Date.now()
                    end = Date.now() + halfAnimationDuration
                    while ((now=Date.now()) < end) {
                        await new Promise((resolve)=>setTimeout(resolve, 0))
                        const newHue = easeInOutQuad(now-start, endHue, startHue-endHue, halfAnimationDuration)
                        const newLightness = easeInOutQuad(now-start, endLightness, startLightness-endLightness, halfAnimationDuration)
                        const newSaturation = easeInOutQuad(now-start, endSaturation, startSaturation-endSaturation, halfAnimationDuration)
                        const newRadius = easeInOutQuad(now-start, endRadius, startRadius-endRadius, halfAnimationDuration)
                        const color = `hsl(${newHue}, ${newLightness}%, ${newSaturation}%)`
                        eachNode.set('stroke', color)
                        eachNode.set('radius', newRadius)
                        canvas.renderAll()
                    }
                    // ensure nothing gets messed up
                    await new Promise((resolve)=>setTimeout(()=>{
                        eachNode.set('stroke', originalColor)
                        eachNode.set('radius', originalRadius)
                        eachNode.set('stroke', `hsl(0, 0%, 82.75%)`)
                        eachNode.set('radius', 50)
                        canvas.renderAll()
                        resolve()
                    }, 100))
                }
                activeAnimations.delete(node.id)
            }
        }
        let prevHoveNodeId
        const hoverLines = []
        const showConnectionsOf = (target)=>{
            // clear previous
            let idBefore = prevHoveNodeId
            let hoverLinesBefore = [...hoverLines]
            for (let each of hoverLinesBefore) {
                canvas.remove(each)
            }
            hoverLines.length = 0
            prevHoveNodeId = null

            if (target) {
                const isHoveringANode = target.type === FabricNode.type.toLowerCase()
                if (isHoveringANode) {
                    const actualTarget = getAllNodes().filter(each=>each.id===target.id)[0]
                    if (actualTarget) {
                        prevHoveNodeId = actualTarget.id
                        for (let each of actualTarget.outputs) {
                            const line = new fabric.Line([...Object.values(getCenter(target)), ...Object.values(getCenter(each))], {
                                objectCaching: false,
                                ...outputConnectionStyles,
                            })
                            hoverLines.push(line)
                            canvas.insertAt(0, line)
                        }
                        
                        for (let each of actualTarget.inputs) {
                            const line = new fabric.Line([...Object.values(getCenter(target)), ...Object.values(getCenter(each))], {
                                objectCaching: false,
                                ...inputConnectionStyles,
                            })
                            hoverLines.push(line)
                            canvas.insertAt(0, line)
                        }
                    }
                }
            }
        }
    return element = FabricCanvas({
        width,
        height,
        backgroundColor,
        selectionColor,
        selectionLineWidth,
        onceFabricLoads:(fabricCanvas)=>{
            canvas = fabricCanvas
            element.timelineManager = new TimelineManager({
                getCurrentState: ()=>canvas.toObject(),
                loadState: (state)=>canvas.loadFromObject(state),
                afterForwardsTimestep: ()=>{
                    const nodes = {}
                    for (let each of getAllNodes()) {
                        if (each.type === FabricNode.type.toLowerCase()) {
                            nodes[each.id] = each
                        }
                    }
                    
                    console.debug(`nodes is:`,Object.fromEntries(Object.entries(nodes).map(([id, node])=>[id, node.energy])))

                    // collect energy from fired nodes
                    const lines = []
                    for (let each of Object.values(nodes)) {
                        if (each.isFiring) {
                            for (const [id, relationship] of Object.entries(each.outputNodeMapping)) {
                                nodes[id].energy += relationship
                                console.debug(`each is:${JSON.stringify({top: each.top, left: each.left, id: each.id})}`)
                            }
                        }
                    }
                    console.debug(`nodes is:`,Object.fromEntries(Object.entries(nodes).map(([id, node])=>[id, node.energy])))

                    // reset nodes that just fired
                    for (let each of Object.values(nodes)) {
                        if (each.isFiring) {
                            each.energy = each.energyAfterFiring
                            each.isFiring = false
                        }
                    }
                    
                    // discover what new nodes are firing
                    for (let each of Object.values(nodes)) {
                        const isAboveThreshold = each.energy >= each.spikeThreshold
                        if (isAboveThreshold) {
                            pulse(each)
                            for (const [id, relationship] of Object.entries(each.outputNodeMapping)) {
                                nodes[id].energy += relationship
                                
                                console.debug(`each is:${JSON.stringify({top: each.top, left: each.left, id: each.id})}`)
                                // show line briefly
                                const line = new fabric.Line([...Object.values(getCenter(each)), ...Object.values(getCenter(nodes[id]))], {
                                    objectCaching: false,
                                    ...outputConnectionStyles,
                                })
                                lines.push(line)
                                canvas.insertAt(0, line)
                            }
                            each.isFiring = true
                        }
                    }
                    setTimeout(() => {
                        for (let each of lines) {
                            canvas.remove(each)
                        }
                    }, 100)
                    
                    // account for decay
                    for (let each of Object.values(nodes)) {
                        if (!each.isFiring) {
                            each.energy -= each.energyDecayRate
                            if (each.energy < each.stableEnergyLevel) {
                                each.energy = each.stableEnergyLevel
                            }
                        }
                    }
                    
                    // update color based on energy
                    for (let each of Object.values(nodes)) {
                        console.debug(`hsl(${energyToHue(each.energy)}, 100%, 50%)`)
                        each.set('fill', `hsl(${energyToHue(each.energy)}, 100%, 50%)`)
                    }
                    console.debug(`nodes is:`,Object.fromEntries(Object.entries(nodes).map(([id, node])=>[id, node.energy])))
                    
                    canvas.renderAll()
                },
            })
            globalThis.canvas = canvas // debugging only
            // TODO: add pan/zoom
            onceFabricLoads&&onceFabricLoads(canvas)
        },
        ...props,
        // onObjectModified: (event)=>{
        //     const { target } = event
        //     if (target?.id) {
        //         const realObject = canvas.objects.filter(each=>each.id===target.id)[0]
        //         Object.assign(realObject,target)
        //     }
        //     canvas.renderAll()
        // },
        onMouseMove: (event)=>{
            const { target } = event
            // not sure why this is needed, but it is (some sort of caching issue with fabric.js) desyncs canvas.objects with the actual visual objects
            if (target?.id) {
                const realObject = getAllNodes().filter(each=>each.id===target.id)[0]
                Object.assign(realObject,target)
                console.debug(`realObject is:${JSON.stringify({top: realObject.top, left: realObject.left, id: realObject.id, stroke: target.stroke})}`,)
            }
            showConnectionsOf(target)
            // canvas.renderAll()
        },
        onMouseDown: (event)=>{
            prevMouseDownTime = Date.now()
            const { target } = event
            if (target) {
                if (target.type === FabricNode.type.toLowerCase()) {
                    
                }
            }
        },
        onMouseUp: (event)=>{
            const countsAsAClick = Date.now()-prevMouseDownTime < 100
            if (countsAsAClick) {
                const { target } = event
                if (target) {
                    if (target.type === FabricNode.type.toLowerCase()) {
                        pulse(target)
                        // TODO: for some reason even .set(key, value) doesn't "stick" when getting the same value inside of afterForwardsTimestep
                        // this is a workaround for that, and eventually this workaround should be removed
                        const actualTarget = getAllNodes().filter(each=>each.id===target.id)[0]
                        // show its prepped
                        actualTarget.set('fill', `hsl(${energyToHue(1)}, 100%, 50%)`)
                        canvas.renderAll()
                        element.timelineManager.scheduleTask(()=>{
                            // actualTarget.willFireNextTimestepBecauseClick = true
                            actualTarget.energy += 1
                        })
                    }
                }
            }
        },
        jsonObjects,
    })
}


// reference material (probably delete after debugging)
// note the .on("modifyPath"), this.controls, and _render() are the interesting bits
    // import * as fabric from "fabric"
    // import rough from "roughjs"
    // import { ArrowHeadStyle } from "./constants"
    // import { getLineAngle } from "./roughutil"

    // export class FabricRoughArrow extends fabric.Path {
    //     static get type() {
    //         return "FabricRoughArrow"
    //     }
    //     constructor(path, options = {}) {
    //         super(path, options)
    //         this.name = "Arrow"
    //         this.points = options.points
    //         this.roughOptions = options.roughOptions
    //         this.roughOptions.seed = this.roughOptions?.seed ?? Math.random() * 100
    //         this.roughGenerator = rough.generator()
    //         this.left = this.left !== 0 ? this.left : options.points[0]
    //         this.top = this.top !== 0 ? this.top : options.points[1]
    //         this.startArrowHeadStyle = options.startArrowHeadStyle || ArrowHeadStyle.NoHead
    //         this.endArrowHeadStyle = options.endArrowHeadStyle || ArrowHeadStyle.Head
    //         this._updateRoughArrow()
    //         this.controls = fabric.controlsUtils.createPathControls(this, {
    //             controlPointStyle: {
    //                 controlStroke: "slateblue",
    //                 controlFill: "slateblue",
    //             },
    //             pointStyle: {
    //                 controlStroke: "slateblue",
    //                 controlFill: "slateblue",
    //             },
    //         })
    //         this.on("modifyPath", () => {
    //             this.editing = true
    //             this._updateRoughArrow()
    //             this.editing = false
    //         })
    //     }

    //     _createPathData(points) {
    //         const [x1, y1, x2, y2] = points
    //         return `M ${x1} ${y1} Q 0 0 ${x2} ${y2}`
    //     }

    //     _updateRoughArrow() {
    //         if (this.isDrawing) {
    //             const [x1, y1, x2, y2] = this.points
    //             const points = [
    //                 { x: x1, y: y1 },
    //                 { x: x2, y: y2 },
    //             ]

    //             const bounds = fabric.util.makeBoundingBoxFromPoints(points)

    //             const widthSign = x2 >= x1 ? 1 : -1
    //             const heightSign = y2 >= y1 ? 1 : -1

    //             const originX = widthSign < 0 ? "right" : "left"
    //             const originY = heightSign < 0 ? "bottom" : "top"
    //             const relativeCenter = this.getRelativeCenterPoint()
    //             const constraint = this.translateToOriginPoint(relativeCenter, originX, originY)
    //             this.set({
    //                 width: Math.abs(bounds.width),
    //                 height: Math.abs(bounds.height),
    //             })
    //             this.setPositionByOrigin(constraint, originX, originY)

    //             const pathData = this._createPathData([(-bounds.width / 2) * widthSign, (-bounds.height / 2) * heightSign, (bounds.width / 2) * widthSign, (bounds.height / 2) * heightSign])

    //             this.path = fabric.util.parsePath(pathData)
    //         } else {
    //             const scaledPath = [
    //                 [this.path[0][0], this.path[0][1] - this.pathOffset.x, this.path[0][2] - this.pathOffset.y],
    //                 [this.path[1][0], this.path[1][1] - this.pathOffset.x, this.path[1][2] - this.pathOffset.y, this.path[1][3] - this.pathOffset.x, this.path[1][4] - this.pathOffset.y],
    //             ]
    //             this.roughArrow = this.roughGenerator.path(fabric.util.joinPath(scaledPath), this.roughOptions)
    //             this.setCoords()

    //             // Use path[0] for start point and path[1] for end point
    //             const [, x1, y1] = this.path[0]
    //             const [, x, y, x2, y2] = this.path[1]
    //             const angleStart = getLineAngle(x - x1, y - y1)
    //             const angleEnd = getLineAngle(x2 - x, y2 - y)
    //             this._updateArrowHeads(x1 - this.pathOffset.x, y1 - this.pathOffset.y, x2 - this.pathOffset.x, y2 - this.pathOffset.y, angleStart, angleEnd)
    //             return
    //         }

    //         this.roughArrow = this.roughGenerator.path(fabric.util.joinPath(this.path), this.roughOptions)

    //         const [, x1, y1, x2, y2] = this.path[1]
    //         const angleStart = getLineAngle(x1 - this.pathOffset.x, y1 - this.pathOffset.y)
    //         const angleEnd = getLineAngle(x2 - this.pathOffset.x, y2 - this.pathOffset.y)

    //         this._updateArrowHeads(x1, y1, x2, y2, angleStart, angleEnd)
    //     }

    //     _updateArrowHeads(x1, y1, x2, y2, angleStart, angleEnd) {
    //         if (this.endArrowHeadStyle !== ArrowHeadStyle.NoHead) {
    //             const isFilled = this.endArrowHeadStyle === ArrowHeadStyle.FilledHead
    //             const headPath = this._calculateHeadPath(x2, y2, angleEnd)
    //             this.endArrowHead = this.roughGenerator.path(headPath + (isFilled ? "Z" : ""), {
    //                 ...this.roughOptions,
    //                 fill: isFilled ? this.roughOptions.stroke : "transparent",
    //             })
    //         }

    //         if (this.startArrowHeadStyle !== ArrowHeadStyle.NoHead) {
    //             const isFilled = this.startArrowHeadStyle === ArrowHeadStyle.FilledHead
    //             const headPath = this._calculateHeadPath(x1, y1, angleStart + Math.PI)
    //             this.startArrowHead = this.roughGenerator.path(headPath + (isFilled ? "Z" : ""), {
    //                 ...this.roughOptions,
    //                 fill: isFilled ? this.roughOptions.stroke : "transparent",
    //             })
    //         }
    //     }

    //     _calculateHeadPath(x, y, angle, headlen = 30) {
    //         const x1 = x - headlen * Math.cos(angle - Math.PI / 6)
    //         const y1 = y - headlen * Math.sin(angle - Math.PI / 6)
    //         const x2 = x - headlen * Math.cos(angle + Math.PI / 6)
    //         const y2 = y - headlen * Math.sin(angle + Math.PI / 6)

    //         return `M ${x1.toFixed(2)} ${y1.toFixed(2)} L ${x.toFixed(2)} ${y.toFixed(2)} L ${x2.toFixed(2)} ${y2.toFixed(2)}`
    //     }

    //     _render(ctx) {
    //         ctx.save()
    //         ctx.lineCap = "round"
    //         const roughCanvas = rough.canvas(ctx.canvas)
    //         roughCanvas.draw(this.roughArrow)
    //         if (this.endArrowHeadStyle !== ArrowHeadStyle.NoHead) roughCanvas.draw(this.endArrowHead)
    //         if (this.startArrowHeadStyle !== ArrowHeadStyle.NoHead) roughCanvas.draw(this.startArrowHead)
    //         ctx.restore()
    //     }

    //     setPoints(points) {
    //         this.points = points
    //         this._updateRoughArrow()
    //     }

    //     update() {
    //         this._updateRoughArrow()
    //         this.dirty = true
    //     }

    //     toObject(propertiesToInclude) {
    //         return {
    //             ...super.toObject(propertiesToInclude),
    //             editing: this.editing,
    //             path: this.path,
    //             points: this.points,
    //             roughOptions: this.roughOptions,
    //             startArrowHeadStyle: this.startArrowHeadStyle,
    //             endArrowHeadStyle: this.endArrowHeadStyle,
    //         }
    //     }
    // }

    // fabric.classRegistry.setClass(FabricRoughArrow)
    // fabric.classRegistry.setSVGClass(FabricRoughArrow)
