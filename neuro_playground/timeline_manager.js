export class TimelineManager {
    constructor({getCurrentState, loadState, afterForwardsTimestep}={}) {
        this.currentTimestep = 0
        this._timeline = []
        this._savedStates = []
        this.getCurrentState = getCurrentState
        this.loadState = loadState
        this.afterForwardsTimestep = afterForwardsTimestep
        try {
            // init this._savedStates[0]
            Promise.resolve(getCurrentState()).then(r=>{this._savedStates[this.currentTimestep] = r})
        } catch (error) {
            // todo: do something better here
            console.warn(`error in getCurrentState:`, error)
        }
    }
    scheduleTask(func, relativeTimeIndex=1) {
        let sourceOfFunction
        try {
            throw Error(``)
        } catch (error) {
            sourceOfFunction = error.stack
            console.debug(`sourceOfFunction is:`,sourceOfFunction)
        }
        this._timeline[this.currentTimestep+relativeTimeIndex] = this._timeline[this.currentTimestep+relativeTimeIndex] || []
        this._timeline[this.currentTimestep+relativeTimeIndex].push([func, sourceOfFunction])
    }
    goForward() {
        this.currentTimestep += 1
        let possiblePromises = []
        for (let [func, sourceOfFunction] of (this._timeline[this.currentTimestep]||[])) {
            try {
                let output = func()
                if (output instanceof Promise) {
                    possiblePromises.push(output.catch(error=>{
                        console.warn(`error in promise oscheduledFuctionf :`, sourceOfFunction)
                    }))
                }
            } catch (error) {
                console.warn(`error in callback of scheduledFuction:`, sourceOfFunction)
            }
        }
        const timestep = this.currentTimestep
        return Promise.all(possiblePromises).then(this.getCurrentState).then((saveState)=>{
            this._savedStates[timestep] = saveState
            if (this.afterForwardsTimestep) {
                return this.afterForwardsTimestep()
            }
        })
    }
    goBackwards() {
        let targetIndex = this.currentTimestep - 1
        let indexCopy = this.currentTimestep
        // get closest in-the-past state
        while (indexCopy >= 0) {
            indexCopy -= 1
            if (this._savedStates[indexCopy]) {
                break
            }
        }
        if (indexCopy < 0) {
            throw Error(`no previous state found`)
        }
        let activeState = this._savedStates[indexCopy]
        return this.loadState(activeState).then(async ()=>{
            // if we had to go futher back than 1, then replace the steps to get back to the target index
            while (indexCopy < targetIndex) {
                indexCopy += 1
                await this.goForward()
            }
        })
    }
}