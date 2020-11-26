<template>
  <div>
    <CheckboxField :checked="parentChecked" :label="label" @value-change="parentChange"/>
    <div class="children" v-show="children">
      <CheckboxField
        v-for="(child, index) of children" :key="child"
        :label="child" :checked="checkedChildren[index]"
        @value-change="childChange(index, $event)"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Prop, Vue } from 'vue-property-decorator';
import Component from 'vue-class-component';
import CheckboxValue from '@/baklava/CheckboxValue';
import CheckboxInput from '@/baklava/input/CheckboxInput.vue';
import CheckboxField from '@/components/CheckboxField.vue';

@Component({
  components: { CheckboxField, CheckboxInput },
})
export default class CheckboxList extends Vue {
  @Prop({ required: true }) label!: string;
  @Prop() children?: [string];
  private parentChecked = CheckboxValue.CHECKED;
  private checkedChildren: Array<CheckboxValue> = [];

  private change(newVal: CheckboxValue) {
    this.parentChecked = newVal;
  }

  private parentChange(newVal: CheckboxValue) {
    this.parentChecked = newVal;
    for (let i = 0; i < this.checkedChildren.length; i += 1) {
      this.$set(this.checkedChildren, i, newVal);
    }
  }

  private childChange(index: number, newVal: CheckboxValue) {
    this.$set(this.checkedChildren, index, newVal);
    this.parentChecked = this.checkedChildren.every((v) => newVal === v)
      ? newVal : CheckboxValue.HALFCHECKED;
  }

  created() {
    if (this.children) {
      this.checkedChildren = new Array(this.children.length).fill(CheckboxValue.CHECKED);
    }
  }
}
</script>

<style scoped>
.children {
  padding-left: 1.5rem;
}
</style>
