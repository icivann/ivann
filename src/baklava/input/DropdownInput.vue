<template>
  <div>
    <div class="button d-sm-flex" @click="toggleOpen">
      {{value}}
      <div class="icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="6" height="4"
             viewBox="0 0 7 4">
          <line x1="0.5" y2="3" x2="3.5" fill="none" stroke="#202020" stroke-width="1"/>
          <line x1="6" x2="3" y2="3" fill="none" stroke="#202020" stroke-width="1"/>
        </svg>
      </div>
    </div>
    <div class="dropdown-container" v-if="open">
      <div class="dropdown-option"
           v-for="item in items"
           :key="item"
           :class="item === value && 'dropdown-selected'"
           @click="select(item)">{{item}}
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class DropdownInput extends Vue {
  @Prop() value!: string;

  @Prop() items!: [string];

  private open = false;

  toggleOpen() {
    this.open = !this.open;
  }

  select(selected: string) {
    this.$emit('value-change', selected);
    this.toggleOpen();
  }
}
</script>

<style scoped>
  .button {
    margin: 0 5px;
    padding: 0 0 0 0.3em;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
    height: 1rem;
  }

  .button:hover {
    background: #e0e0e0;
  }

  .icon {
    padding: 0 0.3em 0.2em 0.4em;
  }

  .dropdown-container {
    position: absolute;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
    margin: 0 5px;
    z-index: 1;
  }

  .dropdown-option {
    width: 100%;
    z-index: 2;
    padding: 0 0.4em 2px 0.3em;
    border-radius: 2px;
  }

  .dropdown-option:hover {
    background: #e0e0e0;
  }

  .dropdown-selected {
    background: #2773ec;
    color: #e0e0e0;
  }

  .dropdown-selected:hover {
    background: #206edc;
  }

</style>
